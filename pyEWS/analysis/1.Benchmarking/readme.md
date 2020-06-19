# Bechmarking EWS moddeling approaches 

The scripts contained in this folder are based on the work i've done in the ../ folder. The
aim is to create a test suite so i can directly compare the results of different modeling 
approaches.  To do this I am creating a series of test datasets that all the models will have 
to deal with and a set of standard tools for recording metadata about the scripts.  


## 1. The standard datasets

Built using test train split.  

	1.1 The train componesnts can be used by both stages, but the test can only be used 
	for the final testing.  
	1.2 The dataset will be broken up into segments so there are no skews that come if i vary
	the landsat inclusion window.  
	1.3 for two stange models, the test/train split will be seeded so that it always gets the 
	same values 