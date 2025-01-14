# Local Feature Detection and Matching Implementation
## About The Project
This project implements a simplified version of SIFT (Scale-Invariant Feature Transform) for local feature matching between images, achieving up to 94% accuracy on complex architectural scenes. The implementation includes a Harris corner detector, SIFT-like feature descriptor, and nearest neighbor distance ratio test for matching.

### Key Performance Results
* Notre Dame: 86% accuracy (improved from 48% baseline)
* Mount Rushmore: 94% accuracy (improved from 88% baseline)
* Episcopal Gaudi: 2% accuracy (improved from 0% baseline)

### Key Features
* Harris corner detector with gradient-based approach and Gaussian blur for noise reduction
* SIFT-like descriptor with:
  * 4x4 grid of cells
  * 8-bin histogram of gradient orientations
  * Magnitude-weighted orientation binning
  * Illumination invariance through normalization
* Nearest neighbor distance ratio test with 0.8 threshold

## Implementation Details

### Interest Point Detection
* Sobel filters for vertical and horizontal gradient computation
* Second-moment matrix construction for corner detection
* Corner response calculation using determinant and trace
* Non-maxima suppression for peak isolation
* Gaussian blur for noise reduction
* Border suppression through careful window management
* Tunable α parameter for accuracy optimization

### Feature Description
* Initial Gaussian blur for noise reduction
* Gradient computation using Sobel operators
* Magnitude and orientation calculation (range [0, 2π])
* 4x4 cell grid with 8-bin orientation histograms
* Gradient magnitude weighting
* Two-stage normalization for illumination invariance
* Additional Gaussian blur per cell for enhanced robustness

### Feature Matching
* Euclidean distance computation between feature sets
* Nearest and second-nearest neighbor identification
* Distance ratio threshold (typically 0.8)
* Confidence score calculation based on distance ratio
* Effective filtering of ambiguous matches

## Results
* Notre Dame Performance:
  * Baseline (normalized patches): 48%
  * SIFT-like implementation: 86%
  * Significant improvement in matching accuracy

* Mount Rushmore Performance:
  * Baseline: 88%
  * SIFT-like implementation: 94%
  * More robust to viewpoint changes

* Episcopal Gaudi Performance:
  * Baseline: 0%
  * SIFT-like implementation: 2%
  * Challenging case showing modest improvement

## Usage
```python
python main.py
```

## Contributing
* Follow implementation guidelines
* Maintain time efficiency
* Document all parameter choices

## License
The project was conducted at Korea Advanced Institute of Sciece and Technology (KAIST) and distributed under the MIT License. See `LICENSE` for more information.



