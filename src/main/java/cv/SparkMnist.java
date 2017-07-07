package cv;

import java.io.IOException;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class SparkMnist {
    public static void main(String[] args) throws IOException{
        if( args.length != 4 ){
            System.err.println("Input Format:<imageFilePath> <numEpochs> <modelPath> <numBatch>");
            return;
        }
        SparkConf conf = new SparkConf()
//                            .setMaster("local[*]")
                            .set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator")  //register kryo for nd4j
                            .setAppName("Mnist Java Spark (Java)");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        final String imageFilePath = args[0];
        final int numEpochs = Integer.parseInt(args[1]);
        final String modelPath = args[2];
        final int numBatch = Integer.parseInt(args[3]);
        //
        //
        JavaRDD<DataSet> javaRDDImageTrain = jsc.objectFile(imageFilePath);     //load image data from hdfs
        ParameterAveragingTrainingMaster trainMaster = new ParameterAveragingTrainingMaster.Builder(numBatch)   //weight average service
                                                            .workerPrefetchNumBatches(0)
                                                            .saveUpdater(true)
                                                            .averagingFrequency(5)
                                                            .batchSizePerWorker(numBatch)
                                                            .build();
        int nChannels = 1;
        int outputNum = 10;
        int iterations = 1;
        int seed = 123;
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()  //define lenent
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.1)
                .learningRateScoreBasedDecayRate(0.5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(20)
                        .nOut(50)
                        .stride(2,2)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build())
                 .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true).pretrain(false);

        MultiLayerConfiguration netconf = builder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(netconf);
        net.setListeners(new ScoreIterationListener(1));
        net.init();
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(jsc, net, trainMaster);
        //train the network on Spark
        for( int i = 0; i < numEpochs; ++i ){
            sparkNetwork.fit(javaRDDImageTrain);
            System.out.println("----- Epoch " + i + " complete -----");
            Evaluation evalActual = sparkNetwork.evaluate(javaRDDImageTrain);
            System.out.println(evalActual.stats());
        }
        //save model
        FileSystem hdfs = FileSystem.get(jsc.hadoopConfiguration());
        Path hdfsPath = new Path(modelPath);
        FSDataOutputStream outputStream = hdfs.create(hdfsPath);
        MultiLayerNetwork trainedNet = sparkNetwork.getNetwork();
        ModelSerializer.writeModel(trainedNet, outputStream, true);
    }
}
