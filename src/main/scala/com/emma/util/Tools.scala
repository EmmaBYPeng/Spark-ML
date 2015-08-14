package com.emma.util

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

object Tools {
  
  def parseData(fileName: String, sc: SparkContext): RDD[LabeledPoint] = {
    val data = sc.textFile(fileName)
    
    data.map { line =>
      val parts = line.split('|')
     
      val value = parts(0).toDouble // Labels: HS index value
      
      val numOfFeatures = parts.length - 1 // discard the first two features
      var features = new Array[Double](numOfFeatures)
      for (i <- 1 to numOfFeatures)
        features(i-1) = parts(i).toDouble
      
      LabeledPoint(value, Vectors.dense(features))
    }.cache()
  }
  
  def processData(parsedRawData: RDD[LabeledPoint], sc: SparkContext): RDD[LabeledPoint] = {
    val value = parsedRawData.map { case LabeledPoint(value, features) => value}
    val num = value.count().toInt - 1
    
    val value_new = value.collect()
    val label = new Array[Int](num)
    for (i <- 1 to num) {
      label(i-1) = if (value_new(i) > value_new(i-1)) 1 else 0
    }
    
    val features = parsedRawData.map { case LabeledPoint(value, features) => features}.take(num)
    
    val data = (label zip features) map { case (l, f) => LabeledPoint(l, f)}
    sc.parallelize(data, 8)
  }
  
}