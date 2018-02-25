"""
This experiment attempts to find independent input sites/compartments.
Written by David McDougall, Feb. 2018

This constructs a simple dendrite model and applies a barrage of random
inputs.  It then measures the correlation between the excitement of
synaptic input sites.  Closer together input sites should have a
stronger correlation.


Result: Typical correlation in terms of period:
Periods:    Correlation,    Iz = Alpha ^ -Distance; Alpha = 1 - 1/Period
1/2         90%             0.603465
1           75%             0.364170
1+1/2       55%             0.219764
2           40%             0.132620
3           20%             0.048296
4            8%             0.017588
5            5%             0.006405


Analysis:  Independent compartments do exist in this model.  Their size
is directly proportional to variable DendriteSegment.period.  Where
exactly you draw the line between one compartment and the next is
subjective, since there are always some degrees both of cooperation and
independence.  Assuming that 10% correlation is the maximum allowed
between independent areas, each area has a radius of approximately 3 to
4 periods.
"""

import numpy as np
import random
import matplotlib.pyplot as plt

class CorrelationSampler:
    def __init__(self, dendrite_root):
        self.root = dendrite_root
        self.sample_points = []
        self.samples = []
        self.num_samples = 0

    def random_sample_points(self, num_points):
        all_points = []
        stack = [self.root]
        while stack:
            segment = stack.pop()
            stack.extend(segment.children)
            for index in xrange(len(segment._synapses)):
                all_points.append( (segment, index) )
        
        num_points = int(round(num_points))
        num_points = min(num_points, len(all_points))
        self.sample_points = random.sample(all_points, num_points)
        self.samples = [[] for i in range(num_points)]
        
    def sample(self, thresh = 20):
        # Reject samples which are uniformly inactive.  I think these
        # are throwing off the correlation, with all zero samples.
        if not any(seg.excitement[idx] >= thresh for seg, idx in self.sample_points):
            return
        for sample_index, location in enumerate(self.sample_points):
            segment, index = location
            sample = segment.excitement[index]
            self.samples[sample_index].append(sample)
        self.num_samples += 1

    def path_to(self, loc, cursor=None):
        """
        Returns a list of pairs of (Segment, Distance) where distance
        is the distance along the segment to travel.
        """
        seg, index = loc
        if cursor is None:
            cursor = self.root
        if seg == cursor:
            return [(cursor, index)]
        for child in cursor.children:
            path_from = self.path_to(loc, child)
            if path_from is not None:
                return [(cursor, len(seg._synapses))] + path_from
        return None

    def distance(self, loc_A, loc_B):
        path_a = self.path_to(loc_A)
        path_b = self.path_to(loc_B)
        # Remove the common elements from the paths
        while path_a and path_b and path_a[0] == path_b[0]:
            path_a.pop(0)
            path_b.pop(0)
        
        dist_a = sum(length for seg, length in path_a)
        dist_b = sum(length for seg, length in path_b)
        if path_a and path_b and path_a[0][0] == path_b[0][0]:
            # Special case where one location lays on the path 
            # from the root to the other.
            return abs(dist_a - dist_b)
        else:
            # Locations on opposite ends of fork.
            return dist_a + dist_b

    def seperation(self, loc_A, loc_B, alpha):
        path_a = self.path_to(loc_A)
        path_b = self.path_to(loc_B)
        # Remove the common elements from the paths
        while path_a and path_b and path_a[0] == path_b[0]:
            path_a.pop(0)
            path_b.pop(0)
        
        dist_a = sum(length for seg, length in path_a)
        dist_b = sum(length for seg, length in path_b)
        if path_a and path_b and path_a[0][0] == path_b[0][0]:
            # Special case where one location lays on the path 
            # from the root to the other.
            distance_decay = alpha ** abs(dist_a - dist_b)
            bifurcation_decay = 1.
        else:
            # Locations on opposite ends of fork.
            distance_decay = alpha ** (dist_a + dist_b)
            bifurcation_decay = .5
        # Count the number of bifurcations passed between the locations.
        for seg, length in path_a[:-1]:
            bifurcation_decay *= 1. / len(seg.children)
        for seg, length in path_b[:-1]:
            bifurcation_decay *= 1. / len(seg.children)

        return distance_decay * bifurcation_decay

    def plot(self, period):
        alpha = 1. - 1./period
        dist  = []
        Iz    = []
        X_cor = []
        for i in xrange(len(self.sample_points)):
            for j in xrange(i + 1, len(self.sample_points)):
                # Reject a random fraction of samples.
                pass
                # Measure distance and hypothesized independence.
                loc_a = self.sample_points[i]
                loc_b = self.sample_points[j]
                dist.append(self.distance(loc_a, loc_b) / (period + .0) )
                Iz.append(1. / self.seperation(loc_a, loc_b, alpha))
                # Measure correlations.
                sv1 = self.samples[i]
                sv2 = self.samples[j]
                cor = np.corrcoef(sv1, sv2)[0,1]
                X_cor.append(cor)
        #plt.figure("Correlation vs Separation.")
        plt.title("Correlation vs Separation")
        plt.plot(np.log(Iz)/np.log(10), X_cor, 'o')
        plt.xlabel("-Log10( Separation );  Separation = Alpha ^ Distance")
        plt.ylabel("Correlation of Excitement")
        #plt.show()

if __name__ == '__main__':
    from connections import DendriteSegment, Synapse, SYN_CONNECTED_ACTIVE
    d = DendriteSegment(None, None, None)
    period = 128
    for i in range(1000):
        d._synapses.append(Synapse(d, None, None, 1.0, None))
    cor = CorrelationSampler(d)
    cor.random_sample_points(200)
    for sample in range(100):
        # Set inputs.
        for syn in d._synapses:
            if random.random() < .02:
                syn.state = SYN_CONNECTED_ACTIVE
        # Run dendrite model.
        d.compute(period)
        d.compute_residual(period, thresholds = [13,])
        # Sample and reset.
        cor.sample(thresh = 0)
        d.reset_synapses()
    cor.plot(period)
    plt.show()
