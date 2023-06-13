package examples;

import dataio.CSVInterface;
import dynamics.Evolutionize;
import interpretability.Prediction;
import records.AnyRecord;
import tsetlin.AutomataLearning;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class CreditSuisseExample {


    public CreditSuisseExample() throws IOException, IllegalAccessException {

        CSVInterface csv = new CSVInterface("data/credit_all.txt", 30);

        AnyRecord anyrecord = csv.createRecord();

        String[] fields = anyrecord.getField_names();

        for (int i = 0; i < fields.length; i++) {
            System.out.println(fields[i]);
        }

        ArrayList<AnyRecord> records = csv.getAllRecords();

        for(AnyRecord record : records) {
            System.out.println(record.toString());
        }

        Evolutionize evolution = new Evolutionize(1, 1);
        evolution.initiate(anyrecord, 10);
        for(int i = 0; i < records.size(); i++) {
            evolution.addValue(records.get(i));
        }
        evolution.fit();
        evolution.initiateConvolutionEncoder();


        AutomataLearning model = new AutomataLearning(
                evolution,
                50,
                100,
                2f,
                2,
                .0f);




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


        //set the max numer of literals
        model.setMaxNumberOfLiterals(0);
        model.setNegativeFocusedSampling(false);


        int[][] Xi = new int[train_set.size()][];
        int[] Y = new int[train_set.size()];
        //create samples from train set
        for(int i = 0; i < train_set.size(); i++) {
            AnyRecord r = train_set.get(i);
            evolution.add(r);
            Xi[i] = evolution.get_last_sample();
            Y[i] = r.getLabel();
        }

        for(int e = 0; e < 100; e++) {

            model.fit(Xi, Y);

            int correct = 0;
            for(AnyRecord record : test_set) {

                int mylabel = record.getLabel();
                Prediction pred = model.predict(record);
                correct += (mylabel == pred.getPred_class()) ? 1 : 0;

            }
            model.getAutomaton().printWeights();
            System.out.println("Correct: " + correct + " " + 100f*(1f*correct/(1f*test_set.size())));
            //print the weights in model


        }

    }

    public static void main(String[] args) throws IOException, IllegalAccessException {

        CreditSuisseExample cs = new CreditSuisseExample();

    }


}
