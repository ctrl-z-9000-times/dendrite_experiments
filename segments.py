""" Plots an example synaptic input and excitment along the length of
the dendrite model.  Written by David McDougall, 2018. """

import numpy as np
from random import random
from connections import DendriteSegment, Synapse, SYN_CONNECTED_ACTIVE

nmda_thresh  = 20
learn_thresh = 13
period       = 100
noise        = 0.01   # Input noise, this fraction of input bits are flipped.

if True:
    # Areas seperated by a short distance cooperate.
    inputs = [0]*300 + [1]*15 + [0]*100 + [random() < .40 for _ in range(50)]

elif True:
    # Overwhelmingly active segments are larger than the active areas of
    # synapses on the dendrites, typically active areas extend .5~2
    # diameters beyond the active areas, where diameter is of the size
    # of the synapse high activity area.
    inputs = [0]*500 + [random() < .25 for _ in range(40)]

else:
    # Maximum Possible Value of Excitement.  This is also the total EPSP
    # emmitted by a single synapse integrated over the whole dendrite.
    inputs = [1] * 1000
    print("PSP total excitement: %f"%DendriteSegment.psp_total_excitement(period))

# Setup Dendrite Model.
d = DendriteSegment(None, None, None)
for i in range(1000):
    d._synapses.append(Synapse(d, None, None, 1.0, None))

# Add some random noise to the inputs.
pad_input_with_zeros = len(d._synapses) - len(inputs)
inputs.extend([0] * pad_input_with_zeros)
for i, inp in enumerate(inputs):
    if random() < noise:
        inputs[i] = not inp

# Set inputs.
for syn, inp in zip(d._synapses, inputs):
    if inp:
        syn.state = SYN_CONNECTED_ACTIVE

# Compute results.
d.compute(period)
thresholds = [learn_thresh, nmda_thresh]
matching_events, active_events = d.compute_residual(period, thresholds)
print("Matching Events, Start / End:")
for evt in matching_events:
    for seg, start, end in evt.spans:
        print("%d / %d"%(start, end))
print("Active Events, Start / End:")
for evt in active_events:
    for seg, start, end in evt.spans:
        print("%d / %d"%(start, end))

# Plot.
import matplotlib.pyplot as plt
plt.plot(np.arange(1000), inputs, 'r',
         np.arange(1000), d.excitement, 'b',)
plt.xlabel("Distance along Dendrite Segment", )
plt.ylabel("EPSPs in Red, Excitement is Blue")
plt.axhline(learn_thresh)
plt.axhline(nmda_thresh)
# Show lines where the excitement crosses thresholds.
for evt in matching_events:
    for seg, start, end in evt.spans:
        plt.axvline(start, color='g')
        plt.axvline(end, color='g')
for evt in active_events:
    for seg, start, end in evt.spans:
        plt.axvline(start, color='r')
        plt.axvline(end, color='r')
plt.show()
