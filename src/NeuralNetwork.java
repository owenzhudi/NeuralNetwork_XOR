import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
	private List<Neuron> inputNeurons;
	private List<Neuron> hiddenLayerNeurons;
	private List<Neuron> outputNeurons;
	
	private int inputCount;
	private int hiddenCount;
	private int outputCount;
	
	private List<List<Double>> weightInputHidden;	// weight from input node to hidden node
	private List<List<Double>> weightHiddenOutput;	// weight from hidden node to output node
	
	private List<List<Double>> trainData;
	private List<List<Double>> inputData;
	private List<List<Double>> outputData;
	
	//private double eta;	// learning rate
	private double totalErr;	// total error
	private double firstBatchError;
	private int numBatch = 0;	// number of batches
	
	public NeuralNetwork(int inputCount, int hiddenCount, int outputCount) {
		this.inputCount = inputCount;
		this.hiddenCount = hiddenCount;
		this.outputCount = outputCount;
		//this.eta = eta;
		initWeight();
		initNeuron();
	}
	
	// initialize weight
	private void initWeight() {
		Random rnd = new Random();
		weightInputHidden = new ArrayList<List<Double>>();
		weightHiddenOutput = new ArrayList<List<Double>>();
		for(int i = 0; i < inputCount; i++) {
			List<Double> col = new ArrayList<>();
			for(int j = 0; j < hiddenCount; j++) {
				double w = rnd.nextDouble() * 2 - 1;	// weight: random value from -1 to 1
				col.add(w);
			}
			weightInputHidden.add(col);
		}
		for(int i = 0; i < hiddenCount; i++) {
			List<Double> col = new ArrayList<>();
			for(int j = 0; j < outputCount; j++) {
				double w = rnd.nextDouble() * 2 - 1;
				col.add(w);
			}
			weightHiddenOutput.add(col);
		}
	}
	
	// initialize neuron
	private void initNeuron() {
		inputNeurons = new ArrayList<Neuron>(inputCount);
		hiddenLayerNeurons = new ArrayList<Neuron>(hiddenCount);
		outputNeurons = new ArrayList<Neuron>(outputCount);
		
		for(int i = 0; i < inputCount; i++) {
			inputNeurons.add(new Neuron(Neuron.TYPE_INPUT));
		}
		for(int i = 0; i < hiddenCount; i++) {
			hiddenLayerNeurons.add(new Neuron(Neuron.TYPE_HIDDEN));
		}
		for(int i = 0; i < outputCount; i++) {
			outputNeurons.add(new Neuron(Neuron.TYPE_OUTPUT));
		}
	}
	
	public void setTrainData(List<List<Double>> data) {
		trainData = data;
		setInputData();
		setOutputData();
	}
	
	private void setInputData() {
		inputData = new ArrayList<>();
		for(int i = 0; i < trainData.size(); i++) {
			List<Double> temp = new ArrayList<>();
			for(int j = 0; j < inputCount; j++) {
				temp.add(trainData.get(i).get(j));
			}
			inputData.add(temp);
		}
	}
	
	private void setOutputData() {
		outputData = new ArrayList<>();
		for(int i = 0; i < trainData.size(); i++) {
			List<Double> temp = new ArrayList<>();
			for(int j = inputCount; j < trainData.get(0).size(); j++) {
				temp.add(trainData.get(i).get(j));
			}
			outputData.add(temp);
		}
	}
	
	private void forward(int num) {
		// input neurons
		for(int i = 0; i < inputCount; i++) {
			inputNeurons.get(i).input(inputData.get(num).get(i));
		}
		
		// hidden layer neurons
		for(int i = 0; i < hiddenCount; i++) {
			double val = 0.0;
			for(int j = 0; j < inputCount; j++) {
				val += inputNeurons.get(j).getValue() * weightInputHidden.get(j).get(i);
			}
			hiddenLayerNeurons.get(i).updateValue(val);
		}
		
		// output neurons
		for(int i = 0; i < outputCount; i++) {
			double val = 0.0;
			for(int j = 0; j < hiddenCount; j++) {
				val += hiddenLayerNeurons.get(j).getValue() * weightHiddenOutput.get(j).get(i);
			}
			outputNeurons.get(i).updateValue(val);
		}
	}
	
	private void backPropagation(int num, double eta) {
		// output neurons
		double[] deltaOutput = new double[outputCount];
		double[] deltaHidden = new double[hiddenCount];
		for(int i = 0; i < outputCount; i++) {
			double outputValue = outputNeurons.get(i).getValue();
			deltaOutput[i] = outputValue * (1 - outputValue) * (outputData.get(num).get(i) - outputValue);
		}
		
		for(int i = 0; i < hiddenCount; i++) {
			double hiddenValue = hiddenLayerNeurons.get(i).getValue();
			for(int j = 0; j < outputCount; j++) {
				deltaHidden[i] += hiddenValue * (1- hiddenValue) * deltaOutput[j] * weightHiddenOutput.get(i).get(j);
			}
		}
		
		// update weight between hidden layer and output neurons
		for(int i = 0; i < hiddenCount; i++) {
			for(int j = 0; j < outputCount; j++) {
				double deltaWeight = eta * deltaOutput[j] * hiddenLayerNeurons.get(i).getValue();
				double newWeight = weightHiddenOutput.get(i).get(j) + deltaWeight;
				weightHiddenOutput.get(i).set(j, newWeight);
			}
		}
		
		// update weight between input layer and hidden layer
		for(int i = 0; i < inputCount; i++) {
			for(int j = 0; j < hiddenCount; j++) {
				double deltaWeight = eta * deltaHidden[j] * inputNeurons.get(i).getValue();
				double newWeight = weightInputHidden.get(i).get(j) + deltaWeight;
				weightInputHidden.get(i).set(j, newWeight);
			}
		}
	}
	
	public void train(double eta, double error) {
		totalErr = Double.MAX_VALUE;
		
		while(totalErr >= error) {
			numBatch++;
			totalErr = 0;
			for(int i = 0; i < trainData.size(); i++) {
				forward(i);
				backPropagation(i, eta);
				// computer total error of the output neurons
				for(int j = 0; j < outputCount; j++) {
					totalErr += Math.pow(outputNeurons.get(j).getValue() - outputData.get(i).get(j), 2) / 2 / 4;
				}
				
				if(numBatch > 10000 && totalErr < error) {
					break;
				}
				
			}
			
			if(numBatch == 1) {
				firstBatchError = totalErr;
			}
			//System.out.println("error: " + error);
			//System.out.println("total error: " + totalErr);
		}
	}
	
	public void printWeights(boolean isInitial) {
		if(isInitial) {
			System.out.println("Initial weights between the input and hidden layer neurons are:");
		} else {
			System.out.println("Final weights between the input and hidden layer neurons are:");
		}
		for(int i = 0; i < inputCount; i++) {
			for(int j = 0; j < hiddenCount; j++) {
				System.out.print("" + weightInputHidden.get(i).get(j) + " ");
				if((j + 1) % hiddenCount == 0) {
					System.out.println();
				}
			}
		}
		
		if(isInitial) {
			System.out.println("Initial weights between the hidden layer and output neurons are:");
		} else {
			System.out.println("Final weights between the hiddenlayer and output neurons are:");
		}
		for(int i = 0; i < hiddenCount; i++) {
			for(int j = 0; j < outputCount; j++) {
				System.out.print("" + weightHiddenOutput.get(i).get(j) + " ");
				if((j + 1) % outputCount == 0) {
					System.out.println();
				}
			}
		}
	}
	
	public double getError() {
		return totalErr;
	}
	
	public double getFirstBatchError() {
		return firstBatchError;
	}
	
	public int getTotalBatches() {
		return numBatch;
	}
	
	public void reset() {
		initWeight();
		initNeuron();
	}
	
	public void test() {
		for(int i = 0; i < outputData.size(); i++) {
			forward(i);
			for(int j = 0; j < outputCount; j++) {
				System.out.println("output = " + outputNeurons.get(j).getValue());
			}
		}
	}
	
}
