import java.util.Collections;
import java.util.List;

public class PerceptronLearner {
    /**
     * Add bias to a list of PVectors
     * @param patterns patterns to add a bias to
     * @param bias value of bias
     */
    public void addBias(List<PVector> patterns, int bias) {
        for (PVector pattern : patterns) {
            pattern.addCoord(bias);
        }
    }
    /**
     * The method that implements perceptron learning
     *
     * @param positive A list of positive training patterns
     * @param negative A list of negative training patterns
     * @param bias Whether you need a bias for this training set
     * @param maxIterations Max number of iterations that the algorithm is allowed to take
     * @param queries List of points whose classifications form the output
     *
     * @return answer string
     */
    public String execute(List<PVector> positive, List<PVector> negative, Boolean bias, Integer maxIterations, List<PVector> queries)
    {

        PVector weight = null; // Weight we're adjusting
        if (bias) {
            weight = PVector.constant(positive.get(0).size() + 1, 1); // Add bias to weight

            // Add biases to every list, so that dot product computes correctly
            addBias(positive, 1);
            addBias(negative, 1);
            addBias(queries, 1);
        } else {
            weight = PVector.constant(positive.get(0).size(), 1);
        }

        int iterations = 0; // Number of epochs
        boolean change = false;
        while (!change) { // While there is no change to the weight vector
            iterations++;
            change = true;

            // Classify positive training set
            for (PVector pattern : positive) {
                if (weight.dotProduct(pattern) <= 0) {
                    weight = weight.add(pattern);
                    change = false; // Change to weight vector, need another iteration
                }
            }

            // Classify negative training set
            for (PVector pattern : negative) {
                if (weight.dotProduct(pattern) > 0) {
                    weight = weight.subtract(pattern);
                    change = false; // Change to weight vector, need another iteration
                }
            }

            if (iterations >= maxIterations) { // Exceeded max amount of iterations
                return maxIterations.toString();
            }
        }

        // Classify the queries
        String result = iterations + " ";
        for (PVector query : queries) {
            if (weight.dotProduct(query) > 0) { // Over the line
                result += "+";
            } else { // Under the line
                result += "-";
            }
        }

        return result;
    }
}
