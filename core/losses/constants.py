"""Core loss-related constants shared across tasks.

These names are intentionally decoupled from concrete loss implementations so
that ``core.trainer.metrics_trainer`` does not need to import task-specific loss
modules.
"""

from __future__ import annotations

# Optional mapping from registered loss class names to shorter log labels.
LOSS_DISPLAY_NAMES = {
    "CompositeLoss": "Composite",
    "SIL1Loss": "SI-L1",
    "SIL1WavSeTrainLoss": "SI-L1_wav",
    "MRSIRelCspecL2Loss": "MR-SI-Rel",
    "MRSIRelCspecL2SuppressPenaltyLoss": "MR-SI-Rel-Suppress",
    "MRSpecSuppressPenaltyLoss": "MR-Suppress",
    "SISNRLoss": "SI-SNR",
    "L1WavLoss": "L1_wav",
    "MultiResSTFTLoss": "MR-STFT",
    "MultiResL1SpecLoss": "MR-L1-Spec",
    "MultiResFreqLoss": "MR-Freq",
    "ComplexIdealRatioMaskLoss": "cIRM-MSE",
    "MultiBandRelativeContrastiveLoss": "MBRCL",
    "MagnitudePlusComplexL1Loss": "Mag+Complex-L1",
    "FWNegSISDRLoss": "FW-Neg-SISDR",
    "StateCrossEntropyLoss": "State-CE",
    "DeepFilterNetLSNRLoss": "DF-LSNR",
    "DeepFilterNetOverAttenuationLoss": "OverAttn",
    "PESQLoss": "PESQ",
    "STOILoss": "STOI",
    "DeepFilterNetMultiResSpecLoss": "DF-MR-Spec",
    "DeepFilterNetAlphaLoss": "DF-Alpha",
    "DeepFilterNetFullLossWithLSNR": "DF-Full-LSNR",
    "DeepFilterNetLSNRComponent": "DF-LSNR-Comp",
    "DfAlphaLossComponent": "DF-Alpha-Comp",
    "AuralossMultiResSTFTLoss": "Auraloss-MR-STFT",
    "AuralossSDRLoss": "Auraloss-SDR",
    "PLCPAAsymmetricLoss": "PLCPA-Asym",
    "PaseMultiScaleMelLoss": "PASE-Mel",
    "MSELoss": "MSE",
    "SpectralConvergenceLoss": "Spec-Conv",
    "LogMagnitudeSTFTLoss": "LogMag-STFT",
}
