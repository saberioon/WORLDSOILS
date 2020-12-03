

The resampled Soil Spectral Libraries
This package contains ten soil spectral datasets that simulated two current multispectral (Landsat 8 OLI, Sentinel-2 MSI, both 2a and 2b) and two forthcoming hyperspectral (EnMAP, CHIME) satellite sensors. The LUCAS15 and Brazilian soil spectral libraries (SSLs) were utilized to produce two different sets of “resampled” spectra. A first set of spectra was obtained by re-sampling the SSLs, using convolution procedures, to the specific spectral response of the Sentinel-2 (a&b) and Landsat8. This process was carried out using the associated spectral response functions. On the other hand, for EnMAP we utilized Gaussian functions.

#### 1 SATELLITE SENSORS
The main technical specifications of the satellite spectral sensors are summarized, in Table 1.
Table 1: Main technical characteristics of the spaceborne sensors considered in this study. The FWHM column indicates the minimum and the maximum of the sensor's bandwidth
|Sensor|	Spectral bands|	Spectral range|	FHWM (nm)|
|---|-----|---|---|
|Sentinel-2MSI|	12	|9VNIR-3SWIR|	10-60|
|Landsat 8 OLI|	7	|5VNIR-2SWIR| 20-200 |
|EnMAP|	242	|420-2450| 10 |
|CHIME|	295	|400-2500| ~7nm |



The resampling analyses were performed using the hsdar (Lehnert et al., 2019) package developed in R software (R Development Core Team, 2013).

#### 2 DATA PARTITIONS

The LUCAS 2015 and Brazilian datasets were split into calibration (66%) and test (33%) sets, using the Conditioned Latin Hypercube Sampling algorithm (Minasny & McBratney, 2006), which is an effective way to replicate the multivariate distribution of the input space in the calibration set. This will enable the learning algorithms to construct their models using data which are representative in the spectral space, compared to Kennard Stone split.
 ( Check the readme_data_partition_20201130 for plots )

#### 3 DESCRIPTION OF DATASETS

 __Check the readme_data_partition_20201130 for details__

  