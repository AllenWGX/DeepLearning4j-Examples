package common;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/***
 * XOR Problem based on FNN and treat it as a regression problem rather than classification
 */
public class XOR {
    public static void main(String[] args){
        int seed = 1234567;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .seed(seed)
                                .iterations(iterations)
                                .learningRate(0.01)
                                .miniBatch(false)
                                .useDropConnect(false)
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                .list()
                                .layer(0, new DenseLayer.Builder()
                                                        .nIn(2)
                                                        .nOut(2).activation(Activation.RELU)//Activation.IDENTITY will not work
                                                                                            //since non-linear transformation
                                                                                            //is needed here
                                                        .build())
                                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                                         .activation(Activation.IDENTITY)
                                                         .nIn(2).nOut(1).build())
                                .backprop(true).pretrain(false)
                                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();        
        //
        double[][] feature = {{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}};
        double[][] label = {{0.0},{1.0},{1.0},{0.0}};
        //
        INDArray ndFeature = Nd4j.create(feature);
        INDArray ndLabel = Nd4j.create(label);
        //
        DataSet ds = new DataSet(ndFeature, ndLabel);
        System.out.println(model.summary());
//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage(); 
//        model.setListeners(new StatsListener(statsStorage, 1));
//        uiServer.attach(statsStorage);
        for(int i = 1; i <= 500; ++i)model.fit(ds);
        INDArray output = model.output(ndFeature);
        System.out.println(output);
//        uiServer.stop();
        //
    }
}
