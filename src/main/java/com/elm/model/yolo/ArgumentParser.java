package com.elm.kafka.video;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.ParameterException;

public class ArgumentParser {
    public static Arguments parseCliArguments(String[] args) {
        Arguments arguments = new Arguments();
        JCommander jcmdr = new JCommander(arguments);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            throw e;
        }
        return arguments;
    }
}
