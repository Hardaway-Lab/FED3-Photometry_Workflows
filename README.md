
Welcome to Synaptech Suite, a fiber photometry analysis tool developed by Neurophotometrics and MetaCell. This modular pipeline will allow you to process your photometry data without the need to write code, and is broken into 3 main modules; signal processing (01), timestamp alignment (02), and signal pooling (03). They should be run in order. 


Currently, Synaptech Suite only supports data recorded with FP3002 systems and the latest version of Neurophotometrics Bonsai software (0.6.0). To update to the latest NPM packages in Bonsai, please see instructions here: https://static1.squarespace.com/static/60ff345fca665d50e1adc805/t/6509eafc18f7817aef5b5404/1695148796487/Neurophotometrics.0.6.0+Update.pdf


The analysis pipeline consists of 3 separate notebooks designed to be run in order:

01.pipeline_process: this script imports raw data, crops a user-specified number of data points, deinterleaves and plots raw data, curve fits and normalizes data, plots normalized data, and writes out processed data in a universal format.

02.pipeline_align: this step imports timestamp data generated by keydown nodes, arduinos, and behavior cameras and aligns timestamps to photometry data

03.pipeline_pooling: the final step of the pipeline segments data around timestamps of interest, creates peri-event plots, and exports segmented data


If you are new to Synaptech Suite, please see our user guide for step-by-step instructions and more information on each processing step.
https://static1.squarespace.com/static/60ff345fca665d50e1adc805/t/653c10e675b3ad1436896ee1/1698435302608/SynaptechSuite+User+Guide.pdf