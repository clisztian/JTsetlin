package examples;

import tsetlin.AutomataAtomLearning;
import tsetlin.ConvolutionEncoder;

import java.io.File;
import java.io.IOException;

public class MNISTExample {

	public static void main(String[] args) throws IOException {

		ClassLoader classLoader = MNISTExample.class.getClassLoader();
		File file = new File(classLoader.getResource("data/MNISTTraining.txt").getFile());
		
		int num_samples = 59000;
		
		int dimX = 28;
		int dimY = 28;
		int patchX = 10; 
		int patchY = 10;
		
		ConvolutionEncoder myEncoder = new ConvolutionEncoder(dimX, dimY, 1, patchX, patchY);
		int[][] X_encoder = myEncoder.bit_encoder(num_samples, file.getAbsolutePath(), ' ');
		int[] label = myEncoder.getLabels();
		
		System.out.println("Num samples: " + label.length);
		int threshold = 5000;
		int nClauses = 500;
		float S = 10f;
		float max_specificity = 10f;
		int nClasses = 10;

		//MultivariateConvolutionalAutomatonMachine conv = new MultivariateConvolutionalAutomatonMachine(myEncoder, threshold, nClasses, nClauses, max_specificity, true, 0f); 
		AutomataAtomLearning conv = new AutomataAtomLearning(myEncoder, threshold, nClasses, nClauses, max_specificity, true, 0f); 
		
		long start = System.currentTimeMillis();
		for(int i = 0; i < 1000; i++) {		
			int pred = conv.update(X_encoder[i], label[i]);	
			System.out.println(i + " " + pred + " " + label[i]);
		}
		long end = System.currentTimeMillis();
		
		System.out.println((float)(end - start)/1000f);
		

		for(int i = 0; i < 100; i++) {
			int[] local_pred = conv.predict_interpret(X_encoder[1000+i]);
			System.out.println("Class predict: " + local_pred[local_pred.length - 1] + " " + label[1000+i]);
		}
		
		
		for(int i = 0; i < nClauses; i++) {
			
			System.out.println(conv.getMachine(0).tm_action(i, 100) + " " + conv.getMachine(7).tm_action(i, 100) + " " + conv.getMachine(9).tm_action(i, 100));
		}
		
		
		System.out.println(conv.getMachine(0).getState() == conv.getMachine(7).getState());
		System.out.println(conv.getMachine(1).getState() == conv.getMachine(9).getState());
	}
	
	
}