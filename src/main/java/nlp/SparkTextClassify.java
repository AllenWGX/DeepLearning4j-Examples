package nlp;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.text.functions.TextPipeline;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import scala.Tuple2;

import java.util.Map;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;


public class SparkTextClassify {

	public static void main(String[] args) throws Exception {
		if( args.length != 6) {
			System.err.println("Error Parameter");
			return;
		}
		//Spark Context Define
        SparkConf conf = new SparkConf()
        					//.setMaster("local[*]")
        					.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator")  //register kryo for nd4j
        					.setAppName("Text Classify Java Spark (Java)");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        //hyper-param
        final int batchSize = Integer.parseInt(args[0]);
        final int totalEpoch = Integer.parseInt(args[1]);
        final String corpusPath = args[2];
        final String modelSavePath = args[3];
        final String VocabCorpusPath = args[4];
        final String VocabLabelPath = args[5];
        //
        Map<String, Object> TokenizerVarMap = new HashMap<>();
        TokenizerVarMap.put("numWords", 1);     //min count of word appearence
        TokenizerVarMap.put("nGrams", 1);       //language model parameter
        TokenizerVarMap.put("tokenizer", DefaultTokenizerFactory.class.getName());  //tokenizer implemention
        TokenizerVarMap.put("tokenPreprocessor", CommonPreprocessor.class.getName());
        TokenizerVarMap.put("useUnk", true);    //unlisted words will use usrUnk
        TokenizerVarMap.put("vectorsConfiguration", new VectorsConfiguration());
        TokenizerVarMap.put("stopWords", new ArrayList<String>());  //stop words
        Broadcast<Map<String, Object>>  broadcasTokenizerVarMap = jsc.broadcast(TokenizerVarMap);   //broadcast the parameter map
        //
        JavaPairRDD<String,String> javaRDDRawCorpus = jsc.textFile(corpusPath + "/*").mapToPair(line-> {
        	String[] token = line.split("\t");
        	return new Tuple2<>(token[0],token[1]);
        }).cache();
        //
        JavaRDD<String> javaRDDCorpus = javaRDDRawCorpus.values();
        TextPipeline textPipeLineCorpus = new TextPipeline(javaRDDCorpus, broadcasTokenizerVarMap);
        textPipeLineCorpus.tokenize();   //tokenize the corpus
        textPipeLineCorpus.buildVocabCache();                                       //build and get the vocabulary
        textPipeLineCorpus.buildVocabWordListRDD();                                 //build corpus
        Broadcast<VocabCache<VocabWord>> vocabCorpus = textPipeLineCorpus.getBroadCastVocabCache();
        JavaRDD<List<VocabWord>> javaRDDVocabCorpus = textPipeLineCorpus.getVocabWordListRDD(); //get tokenized corpus
        //
        JavaRDD<String> javaRDDLabel = javaRDDRawCorpus.keys();
        TextPipeline textPipeLineLabel = new TextPipeline(javaRDDLabel, broadcasTokenizerVarMap);
        textPipeLineLabel.buildVocabCache();                                       //build and get the vocabulary
        Broadcast<VocabCache<VocabWord>> vocabLabel = textPipeLineLabel.getBroadCastVocabCache();
        JavaRDD<VocabWord> javaRDDVocabLabel = textPipeLineLabel.getVocabWordListRDD().map(lst -> lst.get(0));
        JavaPairRDD<List<VocabWord>,VocabWord> javaPairRDDVocabLabel = javaRDDVocabCorpus.zipWithIndex().mapToPair(tuple -> tuple.swap()).join(
        		javaRDDVocabLabel.zipWithIndex().mapToPair(tuple -> tuple.swap())
        ).values().mapToPair(tuple -> tuple);
        //
        final int VOCAB_SIZE = vocabCorpus.value().numWords();
        final int maxCorpusLength = javaRDDVocabCorpus.map(lst -> lst.size()).max(new Comparator<Integer>() {
			
			@Override
			public int compare(Integer o1, Integer o2) {
				return o2.compareTo(o1);
			}
		});
        final long numLabel = javaRDDVocabLabel.map(vw -> vw.getWord()).distinct().count();
        //
        JavaRDD<DataSet> javaRDDTrainData = javaPairRDDVocabLabel.map(new Function<Tuple2<List<VocabWord>,VocabWord>, DataSet>() {
        	 
            @Override
            public DataSet call(Tuple2<List<VocabWord>, VocabWord> tuple) throws Exception {
                List<VocabWord> listWords = tuple._1;
                VocabWord labelWord = tuple._2;
                INDArray features = Nd4j.create(1, 1, maxCorpusLength);
                INDArray labels = Nd4j.create(1, (int)numLabel, maxCorpusLength);
                INDArray featuresMask = Nd4j.zeros(1, maxCorpusLength);
                INDArray labelsMask = Nd4j.zeros(1, maxCorpusLength);
                int[] origin = new int[3];
                int[] mask = new int[2];
                origin[0] = 0;                        //arr(0) store the index of batch sentence
                mask[0] = 0;
                int j = 0;
                for (VocabWord vw : listWords) {         //traverse the list which store an entire sentence
                    origin[2] = j;
                    features.putScalar(origin, vw.getIndex());
                    //
                    mask[1] = j;
                    featuresMask.putScalar(mask, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
                    ++j;
                }
                //
                int lastIdx = listWords.size();
                int idx = labelWord.getIndex();
                labels.putScalar(new int[]{0,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
                labelsMask.putScalar(new int[]{0,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
                return new DataSet(features, labels, featuresMask, labelsMask);
            }
        });
        //
        MultiLayerConfiguration netconf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .iterations(1)
                .learningRate(0.1)
                .learningRateScoreBasedDecayRate(0.5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(true)
                .l2(5 * 1e-4) 
                .updater(Updater.ADAM)
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(VOCAB_SIZE).nOut(512).activation(Activation.IDENTITY).build())
                .layer(1, new GravesLSTM.Builder().nIn(512).nOut(512).activation(Activation.SOFTSIGN).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(512).nOut(2).build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.recurrent(VOCAB_SIZE))
                .build();
        //
        ParameterAveragingTrainingMaster trainMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
										                .workerPrefetchNumBatches(0)
										                .saveUpdater(true)
										                .averagingFrequency(5)
										                .batchSizePerWorker(batchSize)
										                .build();
		SparkDl4jMultiLayer sparknet = new SparkDl4jMultiLayer(jsc, netconf, trainMaster);
		sparknet.setListeners(Collections.<IterationListener>singletonList(new ScoreIterationListener(1)));
		for( int numEpoch = 0; numEpoch < totalEpoch; ++numEpoch){
			sparknet.fit(javaRDDTrainData);
			Evaluation evaluation = sparknet.evaluate(javaRDDTrainData);
			double accuracy = evaluation.accuracy();
			System.out.println("====================================================================");
			System.out.println("Epoch " + numEpoch + " Has Finished");
			System.out.println("Accuracy: " + accuracy);
			System.out.println("====================================================================");
		}
		//
		MultiLayerNetwork network = sparknet.getNetwork();
		FileSystem hdfs = FileSystem.get(jsc.hadoopConfiguration());
		Path hdfsPath = new Path(modelSavePath);
		if( hdfs.exists(hdfsPath) ){
			hdfs.delete(hdfsPath, true);
		}
		FSDataOutputStream outputStream = hdfs.create(hdfsPath);
		ModelSerializer.writeModel(network, outputStream, true);
		/*---Finish Saving the Model------*/
		VocabCache<VocabWord> saveVocabCorpus = vocabCorpus.getValue();
		VocabCache<VocabWord> saveVocabLabel = vocabLabel.getValue();
		SparkUtils.writeObjectToFile(VocabCorpusPath, saveVocabCorpus, jsc);
		SparkUtils.writeObjectToFile(VocabLabelPath, saveVocabLabel, jsc);
		//
        jsc.stop();

	}

}
