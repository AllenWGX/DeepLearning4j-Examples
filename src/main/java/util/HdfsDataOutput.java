package util;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.util.*;
import org.nd4j.linalg.dataset.DataSet;

import org.apache.spark.*;
import org.apache.spark.api.java.JavaSparkContext;

public class HdfsDataOutput {
    public static void main(String[] args) throws Exception{
        int batchSize = 32;
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);
        
        List<DataSet> listMinistTrain = new ArrayList<>();
        List<DataSet> listMinistTest = new ArrayList<>();
        
        while( mnistTrain.hasNext() ){
            listMinistTrain.add(mnistTrain.next());
        }
        
        while( mnistTest.hasNext() ){
            listMinistTest.add(mnistTest.next());
        }
        
        SparkConf conf = new SparkConf()
                            .setMaster("local[*]")
                            .setAppName("Output RDD Mnist");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        
        jsc.parallelize(listMinistTrain).saveAsObjectFile("mnist_train.data");
        jsc.parallelize(listMinistTest).saveAsObjectFile("mnist_test.data");
        
        jsc.stop();
        jsc.close();
    }
}
