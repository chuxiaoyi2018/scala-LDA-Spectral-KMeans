package main.scala

import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer
// $example on$
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
// $example off$
import org.apache.spark.sql.{DataFrame, SQLContext}
import breeze.linalg._
import breeze.numerics._


object KMeans {

  def main(args: Array[String]): Unit = {
    // Creates a Spark context and a SQL context
    val conf = new SparkConf().setAppName(s"${this.getClass.getSimpleName}").setMaster("local");
    conf.set("spark.testing.memory", "2147480000");
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    println("---------------------------Breeze 创建函数-------------------------------")

    val m1 = Vectors.dense(1.0,2.0,3.0)

    val b1 = ArrayBuffer(
      (1.0, m1),
      (2, Vectors.dense(0.1, 0.1, 0.1)),
      (3, Vectors.dense(0.2, 0.2, 0.2)),
      (4, Vectors.dense(9.0, 9.0, 9.0)),
      (5, Vectors.dense(9.1, 9.1, 9.1)),
      (6, Vectors.dense(9.2, 9.2, 9.2))
    )

    println(b1.getClass.getSimpleName)

    // $example on$
    // Crates a DataFrame
    val dataset: DataFrame = sqlContext.createDataFrame(b1).toDF("id", "features")

    // Trains a k-means model
    val kmeans = new KMeans()
      .setK(2)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
    val model = kmeans.fit(dataset)

    // Shows the result
    println("Final Centers: ")
    model.clusterCenters.foreach(println)
    // $example off$

    sc.stop()
  }
}
// scalastyle:on println