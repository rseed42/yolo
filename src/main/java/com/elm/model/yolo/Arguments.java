package com.elm.kafka.video;

import com.beust.jcommander.Parameter;

public class Arguments {
    @Parameter(names = "-batchSize", description = "Minibatch size")
    public int batchSize = 10;

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    public int numEpochs = 1;

    @Parameter(names = "-trainDir", description = "Parent directory of the training data set")
    public String trainDir = "/Users/g6714/Data/fastai/dogscats/train/";

    @Parameter(names = "-testDir", description = "Parent directory of the training data set")
    public String testDir = "/Users/g6714/Data/fastai/dogscats/valid/";

    @Parameter(names = "-imageWidth", description = "Image width that the network can process as input")
    public int imageWidth = 224;

    @Parameter(names = "-imageHeight", description = "Image height that the network can process as input")
    public int imageHeight = 224;

    @Parameter(names = "-imageChannels", description = "Number of color channels for the images. Default is 3 for RGB.")
    public int imageChannels = 3;

    @Parameter(names = "-parameterFilename", description = "Name of the parameter file to store the learned weights")
    public String parameterFilename = "resnet50.zip";

    @Parameter(names = "-randomSeed", description = "Random seed parameter")
    public int randomSeed = 42;

    @Parameter(names = "-showArchitecture", description = "Display the dl4j model summary, which shows all layers")
    public boolean showArchitecture = false;

    @Parameter(names = "-scorePerIteration", description = "When to show the score (every N iterations). Default is 10.")
    public int scorePerIteration = 10;
}

