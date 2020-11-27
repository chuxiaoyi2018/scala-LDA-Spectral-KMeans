package main.test

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics._
import breeze.linalg._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.xerces.impl.xpath.XPath.Axis

import scala.math.exp
import scala.collection.mutable.ArrayBuffer
// $example on$
// $example off$
import org.apache.spark.sql.SQLContext


object Spectral {

  def compute(center: Array[Double], test: Array[Double]):Double = {
    var error = 0.0
    assert(center.length == test.length)
    for(i <- center.indices){
      error += (center(i)- test(i))*(center(i)- test(i))
    }
    return error
  }

  def euclid_distance(v: Vector[Double]):Double = {
    var result = 0.0
    for(i <- 0 until v.size){
      result += v(i) * v(i)
    }
    return math.sqrt(result)
  }

  def kernel_rbf(vector_1: Array[Double], vector_2: Array[Double]):Double = {
    var weight = 0.0
    var l2_norm = 0.0
    val rho = 1000
    assert(vector_1.size == vector_2.size)
    for(i <- 0 until vector_1.size){
      l2_norm += (vector_1(i) - vector_2(i)) * (vector_1(i) - vector_2(i))
    }

    weight = exp(-l2_norm/(2*rho*rho))
    // println(s"l2_norm: $l2_norm")
    // println(s"Weight: $weight")
    return weight
  }



  def predict(center: Array[Array[Double]], test:Array[Array[Double]]):Array[Int] = {
    // test.length row  test(0).length col
    // 如果不使用new来创建那么长度固定为1
    var label = new Array[Int](test.length)

    for(test_index <- test.indices){
      var min = Double.MaxValue
      var temp = 0.0
      for(center_index <- center.indices){
        temp = compute(center(center_index), test(test_index))
        if(min > temp) {
          min = temp
          label(test_index) = center_index
        }
      }
    }
    return label
  }

  def no_diag_max(eig_matrix: DenseMatrix[Double]):(Double, Int, Int) = {
    var max_value = Double.MinValue
    var row_index = 0
    var col_index = 0
    for(i <- 1 until eig_matrix.rows-1){
      for(j <- i+1 until eig_matrix.cols){
        if(math.abs(eig_matrix(i, j)) > max_value){
          row_index = i
          col_index = j
          max_value = math.abs(eig_matrix(i, j))
        }
      }
    }
    return (max_value, row_index, col_index)
  }

  def build_qr_matrix(theta: Double, length:Int, row_index:Int, col_index:Int):DenseMatrix[Double] = {
    var qr_matrix = diag(DenseVector.ones[Double](length))
    qr_matrix(row_index, row_index) = math.cos(theta)
    qr_matrix(col_index, col_index) = qr_matrix(row_index, row_index)

    qr_matrix(row_index, col_index) = - math.sin(theta)
    qr_matrix(col_index, row_index) = - qr_matrix(row_index, col_index)

    return qr_matrix
  }

  def topK_index(eig_v: Array[Double], k:Int):Array[Int] = {
    var temp = 0.0
    var temp_index = 0
    var index_arr = (for(i <- eig_v.indices) yield i).toArray
    for(i <- eig_v.indices){
      for(j <- 0 until eig_v.length - 1){
        if(eig_v(j) > eig_v(j+1)){
          temp = eig_v(j)
          eig_v(j) = eig_v(j+1)
          eig_v(j+1) = temp

          temp_index = index_arr(j)
          index_arr(j) = index_arr(j+1)
          index_arr(j+1) = temp_index
        }
      }
    }


    return index_arr.take(k)
  }

  def jacobi(laplace: DenseMatrix[Double], reduce_dim: Int):DenseMatrix[Double] = {
    val rho = 0.0001

    var eig_v_matrix = diag(DenseVector.fill(laplace.cols)(1.0))
    var eig_matrix = laplace
    var row_index = 0
    var col_index = 0
    var theta = 0.0
    var temp = (0.1, 0, 0)
    var qr_matrix = DenseMatrix.zeros[Double](laplace.rows, laplace.cols)

    var temp_double = 0.0
    while(temp._1 > rho){
      temp = no_diag_max(eig_matrix)
      row_index = temp._2;col_index = temp._3
      theta = 0.5 * scala.math.atan(2*eig_matrix(row_index, col_index)/(eig_matrix(row_index, row_index) - eig_matrix(col_index, col_index)))
      qr_matrix = build_qr_matrix(theta, laplace.cols, row_index, col_index)
      eig_matrix = qr_matrix.t * eig_matrix * qr_matrix
      temp_double = eig_matrix(row_index, col_index)
      eig_v_matrix = eig_v_matrix * qr_matrix
    }

    val eig_v = (for(i <- 0 until eig_v_matrix.cols) yield eig_v_matrix(i,i)).toArray
    val topK = topK_index(eig_v, reduce_dim)

    eig_v_matrix = eig_v_matrix.t

    var result = DenseMatrix.zeros[Double](reduce_dim, laplace.rows)
    for(i <- 0 until reduce_dim){
      for(j <- 0 until laplace.rows){
        result(i,j) = eig_v_matrix(topK(i),j)
      }
    }

    result = result.t

    var l2_norm = DenseVector.zeros[Double](reduce_dim)
    var temp_v = DenseVector.zeros[Double](laplace.rows)
    for(i <- 0 until reduce_dim){
      temp_v = result(::,i)
      l2_norm(i) = euclid_distance(temp_v)
    }

    for(i <- 0 until laplace.rows){
      for(j <- 0 until reduce_dim){
        result(i,j) = result(i,j)/l2_norm(j)
      }
    }

    return result
  }

  def main(args: Array[String]): Unit = {
    // Creates a Spark context and a SQL context
    val cluster_num = 10
    val reduce_dim = 5

    val conf = new SparkConf().setAppName(s"${this.getClass.getSimpleName}").setMaster("local");
    conf.set("spark.testing.memory", "9147480000");
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val train_path = sc.textFile("E:/App/IDEA/workplace/ml/src/data/save_train_500.txt")
      .filter(_.nonEmpty).map(_.split(" ").map(_.toDouble)).map(Vectors.dense)
    val test_path = sc.textFile("E:/App/IDEA/workplace/ml/src/data/save_train_100.txt")
    val real_label = sc.textFile("E:/App/IDEA/workplace/ml/src/data/save_label_500.txt")
      .filter(_.nonEmpty).map(_.split("\n").map(_.toInt)).collect().flatten


    val ori_data = train_path.collect()

    var adjacency = diag(DenseVector.fill(ori_data.length)(1.0))
    // var adjacency = DenseMatrix.zeros[Double](ori_data.length, ori_data.length)
    for(i <- 0 until (ori_data.length - 1)){
      for(j <- i+1 until ori_data.length){
        adjacency(i, j) = kernel_rbf(ori_data(i).toArray, ori_data(j).toArray)
      }
    }


    adjacency = adjacency.t + adjacency - diag(DenseVector.fill(ori_data.length)(1.0))

    println(s"adjacency:$adjacency")
    var degree = DenseMatrix.zeros[Double](ori_data.length, ori_data.length)
    val sum_adjacency = sum(adjacency(*,::))
    for(i <- ori_data.indices){
      degree(i, i) = sum_adjacency(i)
    }

    println(s"degree:$degree")

    var laplace = degree - adjacency

    println(s"laplace:$laplace")

    var degree_0 = DenseMatrix.zeros[Double](ori_data.length, ori_data.length)
    for(i <- ori_data.indices){
      degree_0(i, i) = pow(degree(i, i), -0.5)
    }

    laplace = degree_0 * laplace * degree_0

    println(s"laplace:$laplace")

    // Jacobi Method
    var reduce_matrix = DenseMatrix.zeros[Double](ori_data.length, reduce_dim)
    reduce_matrix = jacobi(laplace, reduce_dim).t

    println(reduce_matrix)
    println(reduce_matrix.cols)
    println(reduce_matrix.rows)

    val buffer = ArrayBuffer(reduce_matrix(::,0).toArray)
    for(i <- 1 until ori_data.length){
      buffer += reduce_matrix(::,i).toArray
    }
    val rdd = sc.makeRDD(buffer).map(f => Vectors.dense(f)).collect()

    val input = ArrayBuffer((1, rdd(0)))
    for(i <- 1 until ori_data.length){
      input += Tuple2(i+1, rdd(i))
    }

    val dataset: DataFrame = sqlContext.createDataFrame(input).toDF("id", "features")

    // Trains a k-means model
    val kmeans = new KMeans()
      .setK(cluster_num)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
    val model = kmeans.fit(dataset)



    val test_data = test_path.filter(_.nonEmpty).map(_.split(" ").map(_.toDouble)).collect()
    val center = model.clusterCenters.map(_.toArray)

    // val predict_label = predict(center, test_data)
    val predict_label = predict(center, buffer.toArray)


    var counter = 0
    for(index<-0 until cluster_num){
      counter += sc.parallelize((for(i <- real_label.indices) yield if(predict_label(i)==index) real_label(i) else -1).filter(_ != -1)).countByValue.values.max.toInt
    }

    var purity = counter.toDouble/real_label.length.toDouble
    println(s"Max/Total: $counter / ${real_label.length}")
    println(s"Purity: $purity")

    sc.stop()

  }
}
// scalastyle:on println