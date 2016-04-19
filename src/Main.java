import java.util.ArrayList;
import java.util.List;

/*
 * Main class for the neural network
 */
public class Main {

	public static void main(String[] args) {
		NeuralNetwork myNetwork = new NeuralNetwork(2, 2, 1);
		
		// input data
		List<List<Double>> data = new ArrayList<List<Double>>();
		List<Double> input0 = new ArrayList<Double>();
		input0.add(0.0); input0.add(0.0); input0.add(0.0);
		List<Double> input1 = new ArrayList<>();
		input1.add(0.0); input1.add(1.0); input1.add(1.0);
		List<Double> input2 = new ArrayList<>();
		input2.add(1.0); input2.add(0.0); input2.add(1.0);
		List<Double> input3 = new ArrayList<>();
		input3.add(1.0); input3.add(1.0); input3.add(0.0);
		data.add(input0);
		data.add(input1);
		data.add(input2);
		data.add(input3);
		
		myNetwork.setTrainData(data);
		myNetwork.printWeights(true);
		myNetwork.train(0.5, 0.1);
		System.out.println("First batch error: " + myNetwork.getFirstBatchError());
		myNetwork.printWeights(false);
		System.out.println("Final error: " + myNetwork.getError());
		System.out.println("Total number of batches: " + myNetwork.getTotalBatches());
		//myNetwork.test();
		
		
		System.out.println("---------------------------------");
		myNetwork.reset();
		myNetwork.printWeights(true);
		myNetwork.train(1, 0.1);
		System.out.println("First batch error: " + myNetwork.getFirstBatchError());
		myNetwork.printWeights(false);
		System.out.println("Final error: " + myNetwork.getError());
		System.out.println("Total number of batches: " + myNetwork.getTotalBatches());
		
		System.out.println("---------------------------------");
		myNetwork.reset();
		myNetwork.printWeights(true);
		myNetwork.train(0.5, 0.02);
		System.out.println("First batch error: " + myNetwork.getFirstBatchError());
		myNetwork.printWeights(false);
		System.out.println("Final error: " + myNetwork.getError());
		System.out.println("Total number of batches: " + myNetwork.getTotalBatches());
		
		System.out.println("---------------------------------");
		myNetwork.reset();
		myNetwork.printWeights(true);
		myNetwork.train(1, 0.02);
		System.out.println("First batch error: " + myNetwork.getFirstBatchError());
		myNetwork.printWeights(false);
		System.out.println("Final error: " + myNetwork.getError());
		System.out.println("Total number of batches: " + myNetwork.getTotalBatches());
		//myNetwork.test();
		
	}

}
