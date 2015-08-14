package com.emma.gbdt

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils

import com.emma.util.Tools

class GBTClassifier {
 
  def main(args: Array[String]) {
    
    println("Please choose a parameter to tune: ") 
    println("-1 >> Default")
    println("0 >> Number of iterations ")
    println("1 >> Max depth")
    println("2 >> Learning rate")
    println("3 >> Max number of bins")
    
    val tune = readLine(">> Input: ")
      
    val conf = new SparkConf().setAppName("Gradient Boosted Trees")
    val sc = new SparkContext(conf)
    
    // Parse and process raw data.
    val parsedRawData = Tools.parseData("price_data_f5.txt", sc)    
    val parsedData = Tools.processData(parsedRawData, sc)
    
    // Train a GBT model.  
    val boostingStrategy = BoostingStrategy.defaultParams("Classification") // Impurity measure: Gini impurity
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    
    var iterations = Array(100)
    var maxDepth = Array(4, 6, 8, 14, 16)
    var learningRate = Array(1e-4, 1e-3, 0.1, 1)
    var maxBins = Array(32, 64, 128)

    val count = tune match {
      case "-1" => 1
      case "0" => iterations.length
      case "1" => maxDepth.length
      case "2" => learningRate.length
      case _ => maxBins.length
    }
    
    // Default index value for each parameter
    var iter = 0
    var depth = 2
    var rate = 2
    var bin = 1
    
    var accuracy = new Array[Double](count) 
    
    // Get test result from k randomly selected test sets
    val k = 10
    for (i <- 0 to k-1) {
      // Split data into training (90%) and test (10%).
      val splits = parsedData.randomSplit(Array((1-1.0/k), 1.0/k), seed = 11L)
      val training = splits(0).cache()
      val test = splits(1)
      
      for(j <- 0 to count-1) {
        
        tune match {
          case "-1" => iter = j
          case "0" => iter = j
          case "1" => depth = j
          case "2" => rate = j
          case _   => bin = j
        }
      
        boostingStrategy.numIterations = iterations(iter)
        boostingStrategy.treeStrategy.maxDepth = maxDepth(depth)
        boostingStrategy.treeStrategy.maxBins = maxBins(rate)
        boostingStrategy.learningRate = learningRate(bin)
        
        val model = GradientBoostedTrees.train(training, boostingStrategy)
        
       // Evaluate model on test instances and compute test error
        val labelAndPreds = test.map { point =>
          val prediction = model.predict(point.features)
          (point.label, prediction)
        }
        accuracy(j) += labelAndPreds.filter(r => r._1 == r._2).count.toDouble / test.count()
      }
    }
    
    // Average accuracy on k test sets
    val acc_avg = accuracy.map(elem => elem/k)
    
    for (i <- 0 to count-1) {
      tune match {
        case "-1" => println("Accuracy for default paras: " + acc_avg(i))
        case "0" => println("Num iteration = " + iterations(i) + " Accuracy = " + acc_avg(i))
        case "1" => println("Max depth = " + maxDepth(i) + " Accuracy = " + acc_avg(i))
        case "2" => println("Learning rate = " + learningRate(i) + " Accuracy = " + acc_avg(i))
        case _ => println("Max bin = " + maxBins(i) + " Accuracy = " + acc_avg(i))
      }
    }
   
  }
  
}