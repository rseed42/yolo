package com.elm.kafka.video;

import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.VideoRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class Application {
    // Logging config
    private static Logger log = LoggerFactory.getLogger(Application.class);
    private static String parentDir = "/Users/g6714/Data/autobahn/";

    private static DataSetIterator loadData() throws IOException, InterruptedException{
        FileSplit fileSplit = new FileSplit(new File(parentDir));
        VideoRecordReader recordReader = new VideoRecordReader(224, 224);
        recordReader.initialize(fileSplit);
        return new RecordReaderDataSetIterator(recordReader, 256);
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        log.info("=== Kafka Video Streaming Application Example ===");
        Arguments cliArgs = ArgumentParser.parseCliArguments(args);

//        DataSetIterator dataIter = loadData();
//
//        int dataSetCounter = 0;
//        while (dataIter.hasNext()) {
//            DataSet ds = dataIter.next();
//            String fileName = String.format("data/data_%05d.bin", dataSetCounter);
//            log.info("Saving {} records to file: {}", ds.numExamples(), fileName);
//            ds.save(new File(fileName));
//            dataSetCounter++;
//        }

    }
}
