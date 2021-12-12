# RatGPS for Python3
This repository contains the data and code to reproduce the results of "Efficient neural decoding of self-location with a deep recurrent network" (see it in https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006822).

...AND potentially, the code that reporoduces the results in https://towardsdatascience.com/reading-rats-mind-2e0b705d5e08 

To run the MLE code and "Bayesian with memory" code, see `Bayesian` folder (and read its README).

To train Recurrent Neural Networks, you need to run `ratcvfit.py` (located in the `1D` and `2D` folders). How to use this Python script is exemplified in `window_scan.sh`.

All figures from the article are included as .png images, but can also be generated anew by running the following notebooks:

``plots/article_plots.ipynb`` Figures 1 and 3
``2D/results.ipynb`` more figures
``2D/gradients.ipynb`` figures relating to gradients
``2D/Activity_tSNE.ipynb`` figures from SI about using T-sne


