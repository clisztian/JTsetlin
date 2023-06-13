package examples;

import be.cylab.java.roc.Roc;
import dataio.CSVInterface;
import dynamics.Evolutionize;
import interpretability.Prediction;
import records.AnyRecord;
import tsetlin.AutomataLearning;
import utils.Parameters;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class BAFExample {

    public BAFExample() throws IOException, IllegalAccessException {

        CSVInterface csv = new CSVInterface("data/Base_red1.csv", 0);
        AnyRecord anyrecord = csv.createRecord();
        String[] fields = anyrecord.getField_names();


        ArrayList<AnyRecord> records = csv.getAllRecords();

        Evolutionize evolution = new Evolutionize(1, 1);
        evolution.initiate(anyrecord, 10);
        for(int i = 0; i < records.size(); i++) {
            evolution.addValue(records.get(i));
        }
        evolution.fit();
        evolution.initiateConvolutionEncoder();

        //model.printNumberOfLiteralsForEachClause();

        int train_set_size = (int) (records.size() * .7);

        //shuffle the records
        Collections.shuffle(records);

        //create two sets of random records
        ArrayList<AnyRecord> train_set = new ArrayList<AnyRecord>();
        ArrayList<AnyRecord> test_set = new ArrayList<AnyRecord>();

        for(int i = 0; i < train_set_size; i++) {
            AnyRecord record = records.get(i);
            train_set.add(record);
        }
        for(int i = train_set_size; i < records.size(); i++) {
            test_set.add(records.get(i));
        }

        int[][] Xi = new int[train_set.size()][];
        int[] Y = new int[train_set.size()];
        //create samples from train set
        for(int i = 0; i < train_set.size(); i++) {
            AnyRecord r = train_set.get(i);
            evolution.add(r);
            Xi[i] = evolution.get_last_sample();
            Y[i] = r.getLabel();
        }


        ArrayList<Parameters> para = generateParameters(500);
        float prev_accuracy = 0;

        Parameters max_parameters = null;
        
        long start = System.currentTimeMillis();
        for(int i = 0; i < 10; i++) {
            Parameters p = para.get(i);
            System.out.println(i + " " + p.n_clauses + " " + p.threshold + " " + p.s + " " + p.maximum_number_of_literals);



            AutomataLearning model = new AutomataLearning(
                    evolution,
                    p.n_clauses,
                    p.threshold,
                    p.s,
                    2,
                    .1f);

            //set the max numer of literals
            model.setMaxNumberOfLiterals(p.maximum_number_of_literals);
            model.setNegativeFocusedSampling(false);



            int count = 0;
            double[] preds = new double[test_set.size()];
            double[] labels = new double[test_set.size()];
            for(int e = 0; e < 1; e++) {


                long start_fit = System.currentTimeMillis();

                model.fit(Xi, Y);

                long end_fit = System.currentTimeMillis();
                System.out.println("Fit time: " + (end_fit - start_fit)/1000f);

    //            for(int i = 0; i < train_set.size(); i++) {
    //                AnyRecord record = train_set.get(i);
    //
    ////                for(int j = 0; j < record.getValues().length; j++) {
    ////                    System.out.print(record.getValues()[j] + " ");
    ////                }
    ////                System.out.println();
    //
    //                int mylabel = record.getLabel();
    //                Prediction pred = model.update_predict(record, mylabel);
    //                System.out.println("Predicted: " + pred.getPred_class() + " " + mylabel );
    //            }

                int correct = 0;
                long start_predict = System.currentTimeMillis();
                for (AnyRecord record : test_set) {

                    int mylabel = record.getLabel();
                    Prediction pred = model.predict(record);
                    //System.out.println("Predicted: " + pred.getPred_class() + " " + mylabel);
                    correct += (mylabel == pred.getPred_class()) ? 1 : 0;

                    labels[count] = mylabel;
                    preds[count] = pred.getPred_class();
                    count++;

                }

                long end_predict = System.currentTimeMillis();
                System.out.println("Predict time: " + (end_predict - start_predict)/1000f);

                //model.getAutomaton().printWeights();
                //System.out.println("Correct: " + correct + " " + 100f * (1f * correct / (1f * test_set.size())));

                Roc roc = new Roc(preds, labels);


                float accuracy = 100f * (1f * correct / (1f * test_set.size()));
                System.out.println("Accuracy: " + accuracy + " auc: " + (float) roc.computeAUC());
                System.out.println("");



                if(accuracy > prev_accuracy) {
                    max_parameters = p;
                    prev_accuracy = accuracy;
                }

                //print the weights in model

            }

        }


        System.out.println("Max parameters: " + max_parameters.n_clauses + " " + max_parameters.threshold + " " + max_parameters.s + " " + max_parameters.maximum_number_of_literals + " " + prev_accuracy);

        long end = System.currentTimeMillis();
        System.out.println("Time: " + (end - start)/1000f + " seconds");
    }



    public static void main(String[] args) throws IOException, IllegalAccessException {
        new BAFExample();
    }



    public ArrayList<Parameters>  generateParameters(int num) {

        Random rng = new Random();



        ArrayList<Parameters> parameters = new ArrayList<Parameters>();
        for(int i = 0; i < num; i++) {
            int nclauses = 1000 + rng.nextInt(1000);
            int threshold = 1000 + rng.nextInt(1000);

            int max_literals = 20 + rng.nextInt(30);

            parameters.add(new Parameters(nclauses, threshold, 11f + rng.nextFloat()*5f, max_literals));
        }
        return parameters;
    }



}

