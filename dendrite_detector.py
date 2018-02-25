# ----------------------------------------------------------------------
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
import math

from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
from nupic.encoders.date import DateEncoder
from nupic.algorithms.spatial_pooler import SpatialPooler
from temporal_memory import TemporalMemory
from nupic.algorithms import anomaly
from nupic.algorithms import anomaly_likelihood

from nab.detectors.base import AnomalyDetector

# Fraction outside of the range of values seen so far that will be considered
# a spatial anomaly regardless of the anomaly likelihood calculation. This
# accounts for the human labelling bias for spatial values larger than what
# has been seen so far.
SPATIAL_TOLERANCE = 0.05


class DendriteDetector(AnomalyDetector):
  def initialize(self):
    # Keep track of value range for spatial anomaly detection.
    self.minVal = None
    self.maxVal = None

    # Time of day encoder
    self.timeOfDayEncoder = DateEncoder(
        timeOfDay = (21,9.49),
        name='time_enc')
    # RDSE encoder for the time series value.
    minResolution = 0.001
    rangePadding  = abs(self.inputMax - self.inputMin) * 0.2
    minVal        = self.inputMin - rangePadding
    maxVal        = self.inputMax + rangePadding
    numBuckets    = 130
    resolution    = max(minResolution, (maxVal - minVal) / numBuckets)
    self.value_enc = RandomDistributedScalarEncoder(
        resolution = resolution,
        name       = 'value_rdse')

    # Spatial Pooler.
    encodingWidth = self.timeOfDayEncoder.getWidth() + self.value_enc.getWidth()
    self.sp = SpatialPooler(
        inputDimensions            = (encodingWidth,),
        columnDimensions           = (2048,),
        potentialPct               = 0.8,
        potentialRadius            = encodingWidth,
        globalInhibition           = 1,
        numActiveColumnsPerInhArea = 40,
        synPermInactiveDec         = 0.0005,
        synPermActiveInc           = 0.003,
        synPermConnected           = 0.2,
        boostStrength              = 0.0,
        seed                       = 1956,
        wrapAround                 = True,)

    self.tm = TemporalMemory(
        columnDimensions          = (2048,),
        cellsPerColumn            = 32,
        activationThreshold       = 20,
        initialPermanence         = .5,         # Increased to connectedPermanence.
        connectedPermanence       = .5,
        minThreshold              = 13,
        maxNewSynapseCount        = 31,
        permanenceIncrement       = 0.04,
        permanenceDecrement       = 0.008,
        predictedSegmentDecrement = 0.001,
        maxSegmentsPerCell        = 128,
        maxSynapsesPerSegment     = 128,        # Changed meaning. Also see connections.topology[2]
        seed                      = 1993,)

    # Initialize the anomaly likelihood object
    numentaLearningPeriod  = int(math.floor(self.probationaryPeriod / 2.0))
    self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
      learningPeriod       = numentaLearningPeriod,
      estimationSamples    = self.probationaryPeriod - numentaLearningPeriod,
      reestimationPeriod   = 100,)
    
    self.age = 0

  def getAdditionalHeaders(self):
    """Returns a list of strings."""
    return ["raw_score"]

  def handleRecord(self, inputData):
    """
    Argument inputData is {"value": instantaneous_value, "timestamp": pandas.Timestamp}
    Returns a tuple (anomalyScore, rawScore).

    Internally to NuPIC "anomalyScore" corresponds to "likelihood_score"
    and "rawScore" corresponds to "anomaly_score". Sorry about that.
    """

    # Check for spatial anomalies and update min/max values.
    value = inputData["value"]
    spatialAnomaly = False
    if self.minVal != self.maxVal:
      tolerance = (self.maxVal - self.minVal) * SPATIAL_TOLERANCE
      maxExpected = self.maxVal + tolerance
      minExpected = self.minVal - tolerance
      if value > maxExpected or value < minExpected:
        spatialAnomaly = True
    if self.maxVal is None or value > self.maxVal:
      self.maxVal = value
    if self.minVal is None or value < self.minVal:
      self.minVal = value

    # Run the HTM stack.  First Encoders.
    timestamp     = inputData["timestamp"]
    timeOfDayBits = np.zeros(self.timeOfDayEncoder.getWidth())
    self.timeOfDayEncoder.encodeIntoArray(timestamp, timeOfDayBits)
    valueBits = np.zeros(self.value_enc.getWidth())
    self.value_enc.encodeIntoArray(value, valueBits)
    encoding = np.concatenate([timeOfDayBits, valueBits])
    # Spatial Pooler.
    activeColumns = np.zeros(self.sp.getNumColumns())
    self.sp.compute(encoding, True, activeColumns)
    activeColumnIndices = np.nonzero(activeColumns)[0]
    # Temporal Memory and Anomaly.
    predictions      = self.tm.getPredictiveCells()
    predictedColumns = list(self.tm.mapCellsToColumns(predictions).keys())
    self.tm.compute(activeColumnIndices, learn=True)
    activeCells = self.tm.getActiveCells()
    rawScore    = anomaly.computeRawAnomalyScore(activeColumnIndices, predictedColumns)

    # Compute log(anomaly likelihood)
    anomalyScore = self.anomalyLikelihood.anomalyProbability(
      inputData["value"], rawScore, inputData["timestamp"])
    finalScore = logScore = self.anomalyLikelihood.computeLogLikelihood(anomalyScore)

    if spatialAnomaly:
      finalScore = 1.0

    if False:
        if self.age == 0:
            print("Correlation Plots ENABLED.")
        # Plot correlation of excitement versus compartmentalization.
        if False:
            start_age = 1000
            end_age   = 1800
        else:
            start_age = 4000
            end_age   = 7260
        if self.age == start_age:
            import correlation
            import random
            self.cor_samplers = []
            sampled_cells = []
            while len(self.cor_samplers) < 20:
                n = random.choice(xrange(self.tm.numberOfCells()))
                if n in sampled_cells:
                    continue
                else:
                    sampled_cells.append(n)
                neuron = self.tm.connections.dataForCell(n)
                if neuron._roots:
                    c = correlation.CorrelationSampler(neuron._roots[0])
                    c.random_sample_points(100)
                    self.cor_samplers.append(c)
            print("Created %d Correlation Samplers"%len(self.cor_samplers))
        if self.age >= start_age:
            for smplr in self.cor_samplers:
                smplr.sample()
        if self.age == end_age:
            import matplotlib.pyplot as plt
            for idx, smplr in enumerate(self.cor_samplers):
                if smplr.num_samples == 0:
                    print("No samples, plot not shown.")
                    continue
                plt.figure("Sample %d"%idx)
                smplr.plot(period = 64)  # Different value!
            plt.show()

    self.age += 1

    return (finalScore, rawScore)
