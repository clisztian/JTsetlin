package examples;

import dataio.CSVInterface;
import dynamics.Evolutionize;
import interpretability.Prediction;
import javafx.util.Pair;
import records.AnyRecord;
import tsetlin.AutomataLearning;
import utils.MutableInt;
import utils.Parameters;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class SupplyChainBayesian {



    private AnyRecord anyrecord;
    private ArrayList<AnyRecord> records;
    ArrayList<AnyRecord> train_set = new ArrayList<AnyRecord>();
    ArrayList<AnyRecord> test_set = new ArrayList<AnyRecord>();
    ArrayList<Integer> model_classes = new ArrayList<Integer>();
    private int[][] Xi;
    private int[] Y;
    private LinkedHashMap<Integer, String> literal_map;
    public SupplyChainBayesian() {

    }

    public Evolutionize createEvolution(int label) throws IllegalAccessException, IOException {

        //CSVInterface csv = new CSVInterface("data/SupplyToBayesian.csv", label);
        CSVInterface csv = new CSVInterface("data/BayesianCustomerChurn_small.csv", label);
        anyrecord = csv.createRecord();
        records = csv.getAllRecords();
        model_classes = new ArrayList<Integer>();

        Evolutionize evolution = new Evolutionize(1, 1);
        evolution.initiate(anyrecord, 4);


        for(int i = 0; i < records.size(); i++) {
            evolution.addValue(records.get(i));

            if(!model_classes.contains(records.get(i).getLabel())) {
                //System.out.println(records.get(i).getLabel());
                model_classes.add(records.get(i).getLabel());
            }
        }
        evolution.fit();
        evolution.initiateConvolutionEncoder();

        literal_map = evolution.getLiteralMap();

        //system print out literal map
//        for(int i = 0; i < literal_map.size(); i++) {
//            System.out.println(i + " " + literal_map.get(i));
//        }


        System.out.println("Number of classes for " + anyrecord.getLabel_name() + ": " + model_classes.size());


        int train_set_size = (int) (records.size() * .8);

        //shuffle the records
        Collections.shuffle(records);

        //create two sets of random records
        train_set = new ArrayList<AnyRecord>();
        test_set = new ArrayList<AnyRecord>();

        for(int i = 0; i < train_set_size; i++) {
            AnyRecord record = records.get(i);
            train_set.add(record);
        }
        for(int i = train_set_size; i < records.size(); i++) {
            test_set.add(records.get(i));
        }

        Xi = new int[train_set.size()][];
        Y = new int[train_set.size()];
        //create samples from train set
        for(int i = 0; i < train_set.size(); i++) {
            AnyRecord r = train_set.get(i);
            evolution.add(r);
            Xi[i] = evolution.get_last_sample();
            Y[i] = r.getLabel();
        }


        return evolution;
    }


    private void construtDatasets(Evolutionize evolution, float percent) throws IllegalAccessException {

        int train_set_size = (int) (records.size() * .8);

        //shuffle the records
        Collections.shuffle(records);

        //create two sets of random records
        train_set = new ArrayList<AnyRecord>();
        test_set = new ArrayList<AnyRecord>();

        for(int i = 0; i < train_set_size; i++) {
            AnyRecord record = records.get(i);
            train_set.add(record);
        }
        for(int i = train_set_size; i < records.size(); i++) {
            test_set.add(records.get(i));
        }

        Xi = new int[train_set.size()][];
        Y = new int[train_set.size()];
        //create samples from train set
        for(int i = 0; i < train_set.size(); i++) {
            AnyRecord r = train_set.get(i);
            evolution.add(r);
            Xi[i] = evolution.get_last_sample();
            Y[i] = r.getLabel();
        }


    }




    public HashMap<String, MutableInt>  model(Evolutionize evolution) throws IllegalAccessException {

        ArrayList<Parameters> para = generateParameters(500);
        float prev_accuracy = 0;

        Parameters max_parameters = null;

        HashMap<String, MutableInt> globalFeatureStrength = new HashMap<String, MutableInt>();

        for(int i = 0; i < 50; i++) {

            construtDatasets(evolution, .8f);


            Parameters p = para.get(i);
            //System.out.println(i + " " + p.n_clauses + " " + p.threshold + " " + p.s + " " + p.maximum_number_of_literals);


            AutomataLearning model = new AutomataLearning(
                    evolution,
                    p.n_clauses,
                    p.threshold,
                    p.s,
                    model_classes.size(),
                    0f);

            //set the max numer of literals
            model.setMaxNumberOfLiterals(p.maximum_number_of_literals);
            model.setNegativeFocusedSampling(false);

            int count = 0;
            double[] preds = new double[test_set.size()];
            double[] labels = new double[test_set.size()];
            for(int e = 0; e < 5; e++) {


//                for(AnyRecord record : train_set) {
//                    int pred = model.update(record, record.getLabel());
//
////                    Prediction pred2 = model.predict(record);
////
////                    System.out.println("Pred: " + pred + " Label: " + record.getLabel() + " " + pred2.getPred_class() + " " + pred2.getProbability());
//                }


                model.fit(Xi, Y);

                int correct = 0;
                count = 0;
                for (AnyRecord record : test_set) {

                    int mylabel = record.getLabel();
                    Prediction pred = model.predict(record);

                    correct += (mylabel == pred.getPred_class()) ? 1 : 0;

                    //System.out.println(mylabel + " " + pred.getPred_class());

                    labels[count] = mylabel;
                    preds[count] = pred.getPred_class();
                    count++;

                }

                float accuracy = 100f * (1f * correct / (1f * test_set.size()));
                //System.out.println("Accuracy: " + accuracy);

                if(accuracy > prev_accuracy) {
                    max_parameters = p;
                    prev_accuracy = accuracy;
                }
            }

            HashMap<String, Integer> localFeatureStrength = computeGlobalFeatureStrength(model);

            int global_count = 0;
            for(String key : localFeatureStrength.keySet()) {

                if(globalFeatureStrength.containsKey(key)) {
                    globalFeatureStrength.get(key).increment();
                } else {
                    globalFeatureStrength.put(key, new MutableInt());
                }

                global_count++;
                if(global_count > 5) {
                    break;
                }
            }

        }

        //System.out.println("\nGlobal Feature Strength");
        globalFeatureStrength = globalFeatureStrength.entrySet().stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));



        System.out.println("\nMax parameters: " + max_parameters.n_clauses + " " + max_parameters.threshold + " " + max_parameters.s + " " + max_parameters.maximum_number_of_literals + " " + prev_accuracy);


        return globalFeatureStrength;

    }





    public HashMap<String, Integer> computeGlobalFeatureStrength(AutomataLearning model) {

        HashMap<String, Integer> globalFeatureStrength = new HashMap<String, Integer>();

        int[][] weights = model.getWeights();

        for(int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {

                ArrayList<Integer> literals = model.getAutomaton().getMachine(i).getLiteralActionsInClause(j);
                for (int k = 0; k < literals.size(); k++) {
                    if (literals.get(k) < literal_map.size()) {

                        String literal = literals.get(k) < literal_map.size() ?
                                literal_map.get(literals.get(k)).split("-")[0] :
                                literal_map.get(literals.get(k) - literal_map.size()).split("-")[0];

                        if (globalFeatureStrength.containsKey(literal)) {
                            globalFeatureStrength.put(literal, globalFeatureStrength.get(literal) + Math.abs(weights[i][j]));
                        } else {
                            globalFeatureStrength.put(literal, Math.abs(weights[i][j]));
                        }
                    }
                }
            }
        }

        //sort globalFeatureStrength by value
        globalFeatureStrength = globalFeatureStrength.entrySet().stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

//        //print globalFeatureStrength
//        for(String key : globalFeatureStrength.keySet()) {
//            System.out.println(key + " " + globalFeatureStrength.get(key));
//        }

        return globalFeatureStrength;
    }





    public ArrayList<Parameters>  generateParameters(int num) {

        Random rng = new Random();



        ArrayList<Parameters> parameters = new ArrayList<Parameters>();
        for(int i = 0; i < num; i++) {
            int nclauses = 30 + rng.nextInt(70);
            int threshold = rng.nextInt(nclauses);

            int max_literals = rng.nextFloat() < .2 ? 0 : 20 + rng.nextInt(30);

            parameters.add(new Parameters(nclauses, threshold, 7f + rng.nextFloat()*5f, max_literals));
        }
        return parameters;
    }


    public static void main(String[] args) throws IllegalAccessException, IOException {

        SupplyChainBayesian bayesian = new SupplyChainBayesian();


        ArrayList<Pair<String, HashMap<String, MutableInt> >> globalFeatureStrengths = new ArrayList<Pair<String, HashMap<String, MutableInt> >>();

        for(int i = 0; i < 23; i++) {

            Evolutionize evolutionize = bayesian.createEvolution(i);

            HashMap<String, MutableInt> globalFeatureStrength = bayesian.model(evolutionize);

            System.out.println("\nGlobal Feature Strength for " + bayesian.getAnyrecord().getLabel_name());

            int count = 0;
            for (String key : globalFeatureStrength.keySet()) {
                System.out.println(key + " " + globalFeatureStrength.get(key).get());
                count++;
                if(count > 5) {
                    break;
                }
            }
            System.out.println("\n");

            Pair<String, HashMap<String, MutableInt>> pair = new Pair<String, HashMap<String, MutableInt>>(bayesian.getAnyrecord().getLabel_name(), globalFeatureStrength);

            globalFeatureStrengths.add(pair);

        }


    }


    public ArrayList<Pair<String, HashMap<String, MutableInt> >> getGlobalFeatureStrengths() throws IllegalAccessException, IOException {

        ArrayList<Pair<String, HashMap<String, MutableInt> >> globalFeatureStrengths = new ArrayList<Pair<String, HashMap<String, MutableInt> >>();

        for(int i = 0; i < 15; i++) {

            Evolutionize evolutionize = createEvolution(i);

            HashMap<String, MutableInt> globalFeatureStrength = model(evolutionize);

            System.out.println("\nGlobal Feature Strength");

            int count = 0;
            for (String key : globalFeatureStrength.keySet()) {
                System.out.println(key + " " + globalFeatureStrength.get(key).get());
                count++;
                if(count > 8) {
                    break;
                }
            }
            System.out.println("\n");

            Pair<String, HashMap<String, MutableInt>> pair = new Pair<String, HashMap<String, MutableInt>>(getAnyrecord().getLabel_name(), globalFeatureStrength);

            globalFeatureStrengths.add(pair);

        }

        return globalFeatureStrengths;
    }






    public AnyRecord getAnyrecord() {
        return anyrecord;
    }
}
