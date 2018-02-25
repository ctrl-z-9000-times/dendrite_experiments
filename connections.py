# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014-2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
# Modified by David McDougall, 2018.

import numpy as np
from bisect import bisect_left
from collections import defaultdict, deque

from nupic.serializable import Serializable

EPSILON = 0.00001 # constant error threshold to check equality of permanences to
                  # other floats

# Synapse state enumeration:
SYN_PRESYN_QUIET     = 0
SYN_CONNECTED_ACTIVE = 1
SYN_POTENTIAL_ACTIVE = 2


"""
Global variable topology is a tuple of
    (max-roots, max-children, max-length)

max-length is the maximum number of synapses allowed on a DendriteSegment.
The dendritic tree is filled out in a breadth first manner.
"""
topology = (1, 2, 128)

class DendriteEvent(object):
    """ This class represents a "segment" of activity.  It is a
    contiguous area of a dendrite tree which exceeds an excitement
    threshold, such as the NMDA spiking threshold.

    Attribute spans is a tuple of (dendrite, start_index, end_index)
    Where dendrite is an instance of DendriteSegment
    Where start_index and end_index are for dendrite.synapses and
        also dendrite.excitement.  end_index is not included in the span!

    These structures are created by the dendrite.compute() method. """

    __slots__ = ('spans', 'num_synapses',
        'numActivePotentialSynapsesForSegment',)

    def __init__(self):
        self.spans = []
        self.num_synapses = 0
        self.numActivePotentialSynapsesForSegment = 0


    def add(self, segment, index):
        """ Add a synapse-location to this event.
        Must form a contiguous event.
        Call this BEFORE resetting the synapse state. """
        # Retrieve the Synapse's data.
        try:
            synapse = segment._synapses[index]
        except IndexError:
            # Adding an implicit synapse, no Synapse data structure.
            synapse = None
        if synapse is not None:
            self.num_synapses += 1
            if synapse.state != SYN_PRESYN_QUIET:
                self.numActivePotentialSynapsesForSegment += 1
        # Add the location to our internal data structure self.spans.
        spans = self.spans
        # Most recently added span is appended to list, check end first.
        for span in reversed(spans):
            seg, start, end = span
            if seg == segment:
                # Extend span.
                if index == end:
                    span[2] = end + 1
                elif index == start - 1:
                    span[1] = index
                else:
                    raise ValueError("DendriteEvent not contiguous.")
                return
        else:
            # Make event span on this new segment.
            span  = [segment, index, index + 1,]
            if False:
                # Check that event is still contiguous.
                contiguous = False
                if index == 0:
                    for seg, start, end in spans:
                        if segment in seg.children:
                            contiguous = True
                            break
                elif index + 1 == len(segment._synapses):
                    for branch in segment.children:
                        if branch in (sp[0] for sp in spans):
                            contiguous = True
                            break
                if spans and not contiguous:
                    raise ValueError("DendriteEvent not contiguous.")
            spans.append(span)


    @property
    def cell(self):
        return self.spans[0][0].cell


    def size(self):
        return sum(end - start for seg, start, end in self.spans)


    def synapse_iter(self):
        for seg, start, end in self.spans:
            for index in range(start, end):
                yield seg._synapses[index]
        raise StopIteration


    def get_free_synapses(self):
        free_spots = []
        for seg, start, end in self.spans:
            for index in range(start, end):
                if seg._synapses[index] is None:
                    free_spots.append((seg, index))
        return free_spots


class DendriteSegment(object):
    """
    Class containing minimal information to identify a unique segment.
    
    :param cell: (int) Index of the cell that this segment is on.
    
    :param flatIdx: (int) The segment's flattened list index.
    
    :param ordinal: (long) Used to sort segments. The sort order needs to be 
           consistent between implementations so that tie-breaking is consistent 
           when finding the best matching segment.
    """

    __slots__ = ["cell", "flatIdx", "_synapses", "_ordinal",
        "children", "excitement", "residuals",]

    def __init__(self, cell, flatIdx, ordinal):
        self.cell      = cell
        self.flatIdx   = flatIdx
        self._synapses = []
        self._ordinal  = ordinal
        self.children  = []
        self.residuals = None


    @staticmethod
    def psp_total_excitement(period):
        """
        Given a Post Synaptic Potential of magnitude 1.0, determines the
        total excitement incured throughout the dendritic tree.  This is
        a theoretic calculation, for verifying actual behavior against.
        """
        alpha  = 1. - 1./period
        return -2. / np.log(alpha)


    def compute(self, period, preorder_excitement=0):
        """
        """
        length     = len(self._synapses)
        alpha      = 1. - 1. / period
        n_children = float(len(self.children))

        # Preorder, walk down the segment, accumulating excitement from
        # the synapses.  This excitement is headed towards the leaves.
        excitement = [0. for _ in xrange(length)]
        for index, syn in enumerate(self._synapses):
            epsp = 1 if (syn is not None and syn.state == SYN_CONNECTED_ACTIVE) else 0
            preorder_excitement = alpha * preorder_excitement + epsp
            excitement[index] = preorder_excitement

        # Recurse to the child branches.  When one branch returns
        # excitement to be distributed across another branch, call it
        # residual and put it in storage at the end of the parent branch.
        if n_children:
            preorder_excitement_split = preorder_excitement / n_children
        else:
            # Preorder excitement goes away when it reaches the end of
            # the dendrite tree.
            preorder_excitement_split = 0
        self.residuals = residuals = [preorder_excitement_split] * int(n_children)
        postorder_excitement = 0
        for index, branch in enumerate(self.children):
            child_excitement = branch.compute(
                period,
                residuals[index])
            residuals[index] = 0
            # Split excitement headed towards the root between the
            # parent and its siblings.
            if n_children > 1:
                parent_split_pct  = .50
                sibling_split_pct = (1. - parent_split_pct) / (n_children - 1.)
            else:
                # Only child case, all returning excitement goes to parent.
                parent_split_pct  = 1.0
                sibling_split_pct = 0.0
            postorder_excitement += parent_split_pct * child_excitement
            sibling_excitement   = sibling_split_pct * child_excitement
            for sibling_index in xrange(int(n_children)):
                if sibling_index == index:
                    continue
                residuals[sibling_index] += sibling_excitement

        # Postorder, walk back up the segment, accumulating excitement.
        # This excitement is headed towards the Soma.
        for index in reversed(xrange(length)):
            syn  = self._synapses[index]
            epsp = 1 if (syn is not None and syn.state == SYN_CONNECTED_ACTIVE) else 0
            # Don't include this synapses PSP at this location since
            # it's already been accounted for on the preorder-traversal
            # side.  Only consider the tail ends which haven't been
            # accounted for.
            postorder_excitement   *= alpha
            excitement[index]      += postorder_excitement
            postorder_excitement   += epsp    # Will affect next location, but not this one.

        self.excitement = excitement
        return postorder_excitement


    def compute_residual(self, period, thresholds, residual=0, current_events=None, all_events=None):
        """
        Argument thresholds is a list of excitement thresholds to look for.
            This will return a tuple with an entry for each threshold,
            containing a list of DendriteEvent objects.
            Thresholds must be sorted, ascending.
        
        Arguments current_events, all_events should NOT be given; they are used internally for recursion.
            Both all_events and current_events run parallel to the thresholds list.
            all_events[threshold_idx] = list of events which meet that threshold.
            current_events[threshold_idx] = The current event or None, is mutated
                while traversing the dendrite.
        
        Returns all_events
        """
        alpha      = 1. - 1. / period
        n_thresh   = len(thresholds)
        excitement = self.excitement
        children   = self.children
        leaf       = not len(children)
        segment_length = topology[2]

        # Setup for first call into this recursive function.
        if all_events is None:
            all_events     = tuple([] for t in thresholds)
            current_events = [None] * n_thresh

        def event_check(index, excite):
            """ Incorporate the given location into all_events and current_events
            if the excitement exceeds a threshold. """
            for evt_idx, thresh in enumerate(thresholds):
                if excite >= thresh:
                    # Add this synapse location to the current event.
                    cur_evt = current_events[evt_idx]
                    if cur_evt is not None:
                        cur_evt.add(self, index)
                    else:
                        # Make a new event.
                        evt = DendriteEvent()
                        evt.add(self, index)
                        current_events[evt_idx] = evt
                        all_events[evt_idx].append(evt)
                else:
                    # Thresholds are sorted, ascending, so once a threshold
                    # fails, all further thresholds will also fail.
                    for no_event_idx in xrange(evt_idx, n_thresh):
                        current_events[no_event_idx] = None
                    break

        # Distribute the residuals throughout.
        for index in xrange(len(excitement)):
            residual = residual * alpha
            x = excitement[index] + residual
            excitement[index] = x
            event_check(index, x)

        # Distribute the excitement onto any implicit synapses which
        # exist at the tips of the leaves.  These synapses are implicitly
        # None's in the segment._synapses list (but not actually allocated).
        if leaf:
            # Include the preorder-excitement which has not been distributed
            # past the end of the explicit _synapses list.
            residual += excitement[-1]  # TODO: I think this adds the residual twice.
            for index in xrange(len(excitement), segment_length):
                residual = residual * alpha
                event_check(index, residual)
                # Because events are sorted, the following line is equivalent to
                # "if any(v is not None for v in current_events):"
                if current_events[0] is not None:
                    # Create the implicit synapse because it is now part of an event.
                    self._synapses.append(None)
                    excitement.append(residual)
                else:
                    # Implicit synapse with no events happening.  Don't create or
                    # represent this synapse, just ignore it and return.
                    break

        # Recurse.
        if children:
            residual = residual / len(children)
            for branch, branch_residual in zip(children, self.residuals):
                branch_residual = branch_residual + residual
                branch.compute_residual(period, thresholds,
                    branch_residual,
                    current_events[:], all_events)

        return all_events


    def reset_synapses(self):
        # Reset Synapse State.
        for syn in self._synapses:
            if syn is not None:
                syn.state = SYN_PRESYN_QUIET
        for child in self.children:
            child.reset_synapses()


    def find_free_segment(self, required_synapses, max_synapses_per_segment,
        depth=0, buffer=None, free_in_buffer=None):
        """ This finds a suitable empty segment with the requested number
        of free synapse locations contained within the maximum segment size.
        
        If this fails to find an existing spot, it returns the DendriteSegment
        which should become the parent of the new DendriteSegment which the
        caller will need to create.
        
        Returns either DendriteEvent or (DendriteSegment, depth)
        The depth can be discarded, it's used internally to ensure the dendritic
        tree is filled out in a breadth first manner.
        """
        # Search for enough free synapses in this segment.
        if buffer is None:
            buffer         = deque()
            # Use a list so that this variable is mutable, even in an inner function.
            free_in_buffer = [0]

        def add_synapse_to_buffer(segment, index):
            buffer.append((segment, index))
            try:
                synapse = segment._synapses[index]
            except IndexError:
                synapse = None  # Implicit Synapse
            if synapse is None:
                free_in_buffer[0] += 1

            if len(buffer) > max_synapses_per_segment:
                pop_seg, pop_idx = buffer.popleft()
                pop_syn = pop_seg._synapses[pop_idx]
                if pop_syn is None:
                    free_in_buffer[0] -= 1

        for index in xrange(len(self._synapses)):
            add_synapse_to_buffer(self, index)

            if free_in_buffer[0] >= required_synapses:
                evt = DendriteEvent()
                for seg, idx in buffer:
                    evt.add(seg, idx)
                return evt

        if not self.children:
            # Check for free synapses at the tips of the leaf branches,
            # which are not represented but do exist.
            implicit_synapses = 0
            for free_syn_index in xrange(len(self._synapses), topology[2]):
                add_synapse_to_buffer(self, free_syn_index)
                implicit_synapses += 1
                if free_in_buffer[0] >= required_synapses:
                    # Add synapses to the end of this segment.
                    self._synapses.extend( [None] * implicit_synapses )
                    # Create a new DendriteEvent for growing on the dendrite tips.
                    evt = DendriteEvent()
                    for seg, idx in buffer:
                        evt.add(seg, idx)
                    return evt
            # No room to grow, no children, consider making a new branch off
            # of the end of this one.
            return self, depth

        elif len(self.children) < topology[1]:
            # Consider making a child branch off of this branch.
            return self, depth
        else:
            # Recurse.
            nearest_dendrite = None
            nearest_dendrite_depth = None
            for dendrite in self.children:
                free_segment = dendrite.find_free_segment(
                    required_synapses, max_synapses_per_segment,
                    depth + 1, deque(buffer), free_in_buffer[:])

                if isinstance(free_segment, DendriteEvent):
                    return free_segment
                else:
                    leaf, depth = free_segment
                    if (nearest_dendrite_depth is None or
                        depth < nearest_dendrite_depth):
                            nearest_dendrite       = leaf
                            nearest_dendrite_depth = depth
            return nearest_dendrite, nearest_dendrite_depth


    def __eq__(self, other):
        """ Explicitly implement this for unit testing. The flatIdx is not designed
        to be consistent after serialize / deserialize, and the synapses might not
        enumerate in the same order.
        """
        return self is other
        return (self.cell == other.cell and
                (sorted(self._synapses, key=lambda x: x._ordinal) ==
                sorted(other._synapses, key=lambda x: x._ordinal)))



class Synapse(object):
    """
    Class containing minimal information to identify a unique synapse.
    
    :param segment: (Object) Segment object that the synapse is synapsed to.
    
    :param presynapticCell: (int) The index of the presynaptic cell of the 
            synapse.
    
    :param permanence: (float) Permanence of the synapse from 0.0 to 1.0.
    
    :param ordinal: (long) Used to sort synapses. The sort order needs to be 
            consistent between implementations so that tie-breaking is consistent 
            when finding the min permanence synapse.
    """
    
    __slots__ = ["segment", "presynapticCell", "permanence", "state", "_ordinal", "index"]
    
    def __init__(self, segment, index, presynapticCell, permanence, ordinal):
        self.segment         = segment
        self.index           = index
        self.presynapticCell = presynapticCell
        self.permanence      = permanence
        self._ordinal        = ordinal
        self.state           = 0


    #def __eq__(self, other):
    #    """ Explicitly implement this for unit testing. Allow floating point
    #    differences for synapse permanence.
    #    """
    #    return (self.segment.cell == other.segment.cell and
    #            self.presynapticCell == other.presynapticCell and
    #            abs(self.permanence - other.permanence) < EPSILON)



class CellData(object):
  """Class containing cell information. Internal to the Connections."""

  __slots__ = ["_segments", "_roots"]

  def __init__(self):
    self._segments = []
    self._roots = []

  def find_free_segment(self, required_synapses, max_synapses_per_segment):
    """ Searches the dendrite tree for a stretch of free synapses.
    Returns either a DendriteEvent specifying the area which contains the synapses
            Or a DendriteSegment or CellData, which should be given a 
            new child DendriteSegment. """
    if len(self._roots) < topology[0]:
        return self

    nearest_dendrite = None
    nearest_dendrite_depth = None
    for dendrite in self._roots:
        free_segment = dendrite.find_free_segment(required_synapses, max_synapses_per_segment)
        if isinstance(free_segment, DendriteEvent):
            return free_segment
        else:
            leaf, depth = free_segment
            if (nearest_dendrite_depth is None or
                depth < nearest_dendrite_depth):
                    nearest_dendrite       = leaf
                    nearest_dendrite_depth = depth
    return nearest_dendrite


def binSearch(arr, val):
  """ 
  Function for running binary search on a sorted list.

  :param arr: (list) a sorted list of integers to search
  :param val: (int)  a integer to search for in the sorted array
  :returns: (int) the index of the element if it is found and -1 otherwise.
  """
  i = bisect_left(arr, val)
  if i != len(arr) and arr[i] == val:
    return i
  return -1



class Connections(Serializable):
  """ 
  Class to hold data representing the connectivity of a collection of cells. 
  
  :param numCells: (int) Number of cells in collection. 
  """

  def __init__(self, numCells):

    # Save member variables
    self.numCells = numCells

    self._cells = [CellData() for _ in xrange(numCells)]
    self._synapsesForPresynapticCell = defaultdict(set)
    self._segmentForFlatIdx = []

    self._numSynapses = 0
    self._freeFlatIdxs = []
    self._nextFlatIdx = 0

    # Whenever creating a new Synapse or Segment, give it a unique ordinal.
    # These can be used to sort synapses or segments by age.
    self._nextSynapseOrdinal = long(0)
    self._nextSegmentOrdinal = long(0)


  def segmentsForCell(self, cell):
    """ 
    Returns the segments that belong to a cell.

    :param cell: (int) Cell index
    :returns: (list) Segment objects representing segments on the given cell.
    """

    return self._cells[cell]._segments


  def synapsesForSegment(self, segment):
    """ 
    Returns the synapses on a segment.

    :param segment: (int) Segment index
    :returns: (set) Synapse objects representing synapses on the given segment.
    """

    return segment._synapses


  def dataForCell(self, cell_index):
    return self._cells[cell_index]


  def dataForSynapse(self, synapse):
    """ 
    Returns the data for a synapse.

    .. note:: This method exists to match the interface of the C++ Connections. 
       This allows tests and tools to inspect the connections using a common 
       interface.

    :param synapse: (:class:`Synapse`)
    :returns: Synapse data
    """
    return synapse


  def dataForSegment(self, segment):
    """ 
    Returns the data for a segment.

    .. note:: This method exists to match the interface of the C++ Connections. 
       This allows tests and tools to inspect the connections using a common 
       interface.

    :param segment (:class:`Segment`)
    :returns: segment data
    """
    return segment


  def getSegment(self, cell, idx):
    """ 
    Returns a :class:`Segment` object of the specified segment using data from 
    the ``self._cells`` array.

    :param cell: (int) cell index
    :param idx:  (int) segment index on a cell
    :returns: (:class:`Segment`) Segment object with index idx on the specified cell
    """

    return self._cells[cell]._segments[idx]


  def segmentForFlatIdx(self, flatIdx):
    """ 
    Get the segment with the specified flatIdx.

    :param flatIdx: (int) The segment's flattened list index.

    :returns: (:class:`Segment`)
    """
    return self._segmentForFlatIdx[flatIdx]


  def segmentFlatListLength(self):
    """ 
    Get the needed length for a list to hold a value for every segment's 
    flatIdx.

    :returns: (int) Required list length
    """
    return self._nextFlatIdx


  def synapsesForPresynapticCell(self, presynapticCell):
    """ 
    Returns the synapses for the source cell that they synapse on.

    :param presynapticCell: (int) Source cell index

    :returns: (set) :class:`Synapse` objects
    """
    return self._synapsesForPresynapticCell[presynapticCell]


  def createSegment(self, cell, parent):
    """ 
    Adds a new segment on a cell.

    :param cell: (int) Cell index
    :param parent: (DendriteSegment or None)  If None, adds new dendrite root segment.
    :returns: (int) New segment index
    """
    cellData = self._cells[cell]

    if len(self._freeFlatIdxs) > 0:
      flatIdx = self._freeFlatIdxs.pop()
    else:
      flatIdx = self._nextFlatIdx
      self._segmentForFlatIdx.append(None)
      self._nextFlatIdx += 1

    ordinal = self._nextSegmentOrdinal
    self._nextSegmentOrdinal += 1

    segment = DendriteSegment(cell, flatIdx, ordinal)
    cellData._segments.append(segment)
    self._segmentForFlatIdx[flatIdx] = segment

    if isinstance(parent, CellData):
        parent._roots.append(segment)

    elif isinstance(parent, DendriteSegment):
        parent.children.append(segment)

    return segment


  def destroySegment(self, segment):
    """
    Destroys a segment.

    :param segment: (:class:`Segment`) representing the segment to be destroyed.
    """
    1/0 # Don't call this method.

    # Remove the synapses from all data structures outside this Segment.
    for synapse in segment._synapses:
      self._removeSynapseFromPresynapticMap(synapse)
    self._numSynapses -= len(segment._synapses)

    # Remove the segment from the cell's list.
    segments = self._cells[segment.cell]._segments
    i = segments.index(segment)
    del segments[i]

    # Free the flatIdx and remove the final reference so the Segment can be
    # garbage-collected.
    self._freeFlatIdxs.append(segment.flatIdx)
    self._segmentForFlatIdx[segment.flatIdx] = None


  def createSynapse(self, segment, index, presynapticCell, permanence):
    """ 
    Creates a new synapse on a segment.

    :param segment: (:class:`Segment`) Segment object for synapse to be synapsed 
           to.
    :param presynapticCell: (int) Source cell index.
    :param permanence: (float) Initial permanence of synapse.
    :returns: (:class:`Synapse`) created synapse
    """
    if segment._synapses[index] is not None:
        raise ValueError("Can not make synapse at this postsynaptic location, synapse already exists here.")

    synapse = Synapse(segment, index, presynapticCell, permanence,
                      self._nextSynapseOrdinal)

    segment._synapses[index] = synapse

    self._synapsesForPresynapticCell[presynapticCell].add(synapse)

    self._nextSynapseOrdinal += 1
    self._numSynapses += 1
    return synapse


  def _removeSynapseFromPresynapticMap(self, synapse):
    inputSynapses = self._synapsesForPresynapticCell[synapse.presynapticCell]

    inputSynapses.remove(synapse)

    if len(inputSynapses) == 0:
      del self._synapsesForPresynapticCell[synapse.presynapticCell]


  def destroySynapse(self, synapse):
    """
    Destroys a synapse.

    :param synapse: (:class:`Synapse`) synapse to destroy
    """

    self._numSynapses -= 1

    self._removeSynapseFromPresynapticMap(synapse)

    # index = synapse.segment._synapses.index(synapse)
    index = synapse.index
    assert(synapse.segment._synapses[index] == synapse)
    synapse.segment._synapses[index] = None


  def updateSynapsePermanence(self, synapse, permanence):
    """ 
    Updates the permanence for a synapse.
    
    :param synapse: (class:`Synapse`) to be updated.
    :param permanence: (float) New permanence.
    """

    synapse.permanence = permanence


  def computeActivity(self, activePresynapticCells, connectedPermanence,
                      activationThreshold, minThreshold, segmentSize):
    """ 
    Compute each segment's number of active synapses for a given input.
    In the returned lists, a segment's active synapse count is stored at index
    ``segment.flatIdx``.

    :param activePresynapticCells: (iter) Active cells.
    :param connectedPermanence: (float) Permanence threshold for a synapse to be 
           considered connected

    :returns: (pair) (``activation_events`` [list],
                      ``minThreshold_events`` [list])
    """
    active_events = []
    learning_events = []
    permanence_threshold = connectedPermanence - EPSILON
    thresholds = [minThreshold, activationThreshold]
    assert(thresholds == sorted(thresholds))

    for cell in self._cells:
      for dendrite in cell._roots:
        dendrite.reset_synapses()

    for cell in activePresynapticCells:
      for synapse in self._synapsesForPresynapticCell[cell]:
        if synapse.permanence > permanence_threshold:
            synapse.state = SYN_CONNECTED_ACTIVE
        else:
            synapse.state = SYN_POTENTIAL_ACTIVE
    for cell in self._cells:
      for dendrite in cell._roots:
        dendrite.compute(segmentSize)
        learn, actv = dendrite.compute_residual(segmentSize, thresholds)
        active_events.extend(actv)
        learning_events.extend(learn)

    return (active_events, learning_events)


  def numSegments(self, cell=None):
    """ 
    Returns the number of segments.

    :param cell: (int) Optional parameter to get the number of segments on a 
           cell.
    :returns: (int) Number of segments on all cells if cell is not specified, or 
              on a specific specified cell
    """
    if cell is not None:
      return len(self._cells[cell]._segments)

    return self._nextFlatIdx - len(self._freeFlatIdxs)


  def numSynapses(self, segment=None):
    """ 
    Returns the number of Synapses.

    :param segment: (:class:`Segment`) Optional parameter to get the number of 
           synapses on a segment.

    :returns: (int) Number of synapses on all segments if segment is not 
              specified, or on a specified segment.
    """
    if segment is not None:
      return len(segment._synapses)
    return self._numSynapses


  def segmentPositionSortKey(self, segment):
    """ 
    Return a numeric key for sorting this segment. This can be used with the 
    python built-in ``sorted()`` function.

    :param segment: (:class:`Segment`) within this :class:`Connections` 
           instance.
    :returns: (float) A numeric key for sorting.
    """
    return segment.cell # + (segment._ordinal / float(self._nextSegmentOrdinal))


  def write(self, proto):
    """ 
    Writes serialized data to proto object.

    :param proto: (DynamicStructBuilder) Proto object
    """
    1/0 # Unimplemented.
    protoCells = proto.init('cells', self.numCells)

    for i in xrange(self.numCells):
      segments = self._cells[i]._segments
      protoSegments = protoCells[i].init('segments', len(segments))

      for j, segment in enumerate(segments):
        synapses = segment._synapses
        protoSynapses = protoSegments[j].init('synapses', len(synapses))

        for k, synapse in enumerate(sorted(synapses, key=lambda s: s._ordinal)):
          protoSynapses[k].presynapticCell = synapse.presynapticCell
          protoSynapses[k].permanence = synapse.permanence


  @classmethod
  def read(cls, proto):
    """ 
    Reads deserialized data from proto object

    :param proto: (DynamicStructBuilder) Proto object

    :returns: (:class:`Connections`) instance
    """
    1/0 # Unimplemented.
    #pylint: disable=W0212
    protoCells = proto.cells
    connections = cls(len(protoCells))

    for cellIdx, protoCell in enumerate(protoCells):
      protoCell = protoCells[cellIdx]
      protoSegments = protoCell.segments
      connections._cells[cellIdx] = CellData()
      segments = connections._cells[cellIdx]._segments

      for segmentIdx, protoSegment in enumerate(protoSegments):
        segment = Segment(cellIdx, connections._nextFlatIdx,
                          connections._nextSegmentOrdinal)

        segments.append(segment)
        connections._segmentForFlatIdx.append(segment)
        connections._nextFlatIdx += 1
        connections._nextSegmentOrdinal += 1

        synapses = segment._synapses
        protoSynapses = protoSegment.synapses

        for synapseIdx, protoSynapse in enumerate(protoSynapses):
          presynapticCell = protoSynapse.presynapticCell
          synapse = Synapse(segment, presynapticCell, protoSynapse.permanence,
                            ordinal=connections._nextSynapseOrdinal)
          connections._nextSynapseOrdinal += 1
          synapses.add(synapse)
          connections._synapsesForPresynapticCell[presynapticCell].add(synapse)

          connections._numSynapses += 1

    #pylint: enable=W0212
    return connections


  def __eq__(self, other):
    """ Equality operator for Connections instances.
    Checks if two instances are functionally identical

    :param other: (:class:`Connections`) Connections instance to compare to
    """
    #pylint: disable=W0212
    for i in xrange(self.numCells):
      segments = self._cells[i]._segments
      otherSegments = other._cells[i]._segments

      if len(segments) != len(otherSegments):
        return False

      for j in xrange(len(segments)):
        segment = segments[j]
        otherSegment = otherSegments[j]
        synapses = segment._synapses
        otherSynapses = otherSegment._synapses

        if len(synapses) != len(otherSynapses):
          return False

        for synapse in synapses:
          found = False
          for candidate in otherSynapses:
            if synapse == candidate:
              found = True
              break

          if not found:
            return False

    if (len(self._synapsesForPresynapticCell) !=
        len(self._synapsesForPresynapticCell)):
      return False

    for i in self._synapsesForPresynapticCell.keys():
      synapses = self._synapsesForPresynapticCell[i]
      otherSynapses = other._synapsesForPresynapticCell[i]
      if len(synapses) != len(otherSynapses):
        return False

      for synapse in synapses:
        found = False
        for candidate in otherSynapses:
          if synapse == candidate:
            found = True
            break

        if not found:
          return False

    if self._numSynapses != other._numSynapses:
      return False

    #pylint: enable=W0212
    return True


  def __ne__(self, other):
    """ 
    Non-equality operator for Connections instances.
    Checks if two instances are not functionally identical

    :param other: (:class:`Connections`) Connections instance to compare to
    """
    return not self.__eq__(other)
