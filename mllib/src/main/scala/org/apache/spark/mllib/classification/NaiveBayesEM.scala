/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.classification

import java.lang.{Iterable => JIterable}

import org.apache.spark.mllib.classification.NaiveBayes.{Bernoulli, Multinomial, supportedModelTypes}
import org.apache.spark.mllib.linalg.{BLAS, DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkException}

import scala.util.Random


object ArrayUtil {
  def normalize(array: Array[Double]) : Array[Double] = {
    val sum = array.sum
    if (sum != 0) { array.map(v => v / sum) } else { array }
  }

  def times(array: Array[Double], coef: Double) : Array[Double] = {
    array.map(v => v * coef)
  }
}

object NaiveBayesEM {
  def train(input: RDD[LabeledPoint]): NaiveBayesModel = {
    new NaiveBayesEM().run(input)
  }

  def train(input: RDD[LabeledPoint], lambda: Double): NaiveBayesModel = {
    new NaiveBayesEM(lambda, Multinomial).run(input)
  }

  def train(input: RDD[LabeledPoint], lambda: Double, modelType: String): NaiveBayesModel = {
    require(supportedModelTypes.contains(modelType),
      s"NaiveBayes was created with an unknown modelType: $modelType.")
    new NaiveBayesEM(lambda, modelType).run(input)
  }
}

// added by mrikitoku
class NaiveBayesEM private(
    private var lambda: Double,
    private var modelType: String,
    private var maxIter: Int = 10)
  extends Serializable with Logging {

  val tolerance = 1.0e-3
  val trainer = new NaiveBayes(lambda)

  val requireNonnegativeValues: Vector => Unit = (v: Vector) => {
    val values = v match {
      case sv: SparseVector => sv.values
      case dv: DenseVector => dv.values
    }
    if (!values.forall(_ >= 0.0)) {
      throw new SparkException(s"Naive Bayes requires nonnegative feature values but found $v.")
    }
  }

  val requireZeroOneBernoulliValues: Vector => Unit = (v: Vector) => {
    val values = v match {
      case sv: SparseVector => sv.values
      case dv: DenseVector => dv.values
    }
    if (!values.forall(v => v == 0.0 || v == 1.0)) {
      throw new SparkException(
        s"Bernoulli naive Bayes requires 0 or 1 feature values but found $v.")
    }
  }

  def this(lambda: Double) = this(lambda, NaiveBayes.Multinomial)

  def this() = this(1.0, NaiveBayes.Multinomial)

  /** Set the smoothing parameter. Default: 1.0. */
  def setLambda(lambda: Double): NaiveBayesEM= {
    this.lambda = lambda
    this.trainer.setLambda(lambda)
    this
  }

  /** Get the smoothing parameter. */
  def getLambda: Double = lambda

  /**
   * Set the model type using a string (case-sensitive).
   * Supported options: "multinomial" (default) and "bernoulli".
   */
  def setModelType(modelType: String): NaiveBayesEM = {
    require(NaiveBayes.supportedModelTypes.contains(modelType),
      s"NaiveBayes was created with an unknown modelType: $modelType.")
    this.modelType = modelType
    this
  }

  /** Get the model type. */
  def getModelType: String = this.modelType

  def setMaxIter(iter: Int): NaiveBayesEM = {
    this.maxIter = iter
    this
  }

  def getMaxIter : Double = this.maxIter

  def aggregateLabeledData(data: RDD[LabeledPoint]): Array[(Double, (Double, DenseVector))] = {
    val aggregated = this.aggregateLabeledData2(data.map(p => (p.label, p.features)))
    aggregated
  }

  def aggregateLabeledData2(data: RDD[(Double, Vector)]): Array[(Double, (Double, DenseVector))] = {
    val aggregated = data.combineByKey[(Double, DenseVector)](
      createCombiner = (v: Vector) => {
        if (modelType == Bernoulli) {
          requireZeroOneBernoulliValues(v)
        } else {
          requireNonnegativeValues(v)
        }
        (1.0, v.copy.toDense)
      },
      mergeValue = (c: (Double, DenseVector), v: Vector) => {
        requireNonnegativeValues(v)
        BLAS.axpy(1.0, v, c._2)
        (c._1 + 1L, c._2)
      },
      mergeCombiners = (c1: (Double, DenseVector), c2: (Double, DenseVector)) => {
        BLAS.axpy(1.0, c2._2, c1._2)
        (c1._1 + c2._1, c1._2)
      }
    ).collect().sortBy(_._1)
    aggregated
  }

  def aggregateData(data: RDD[(Double, Vector)], dimension: Int): (Double, Vector) = {
    val aggregated = data.aggregate((0.0, new DenseVector(Array.fill(dimension)(0.0))))(
      seqOp = (u: (Double, DenseVector), v: (Double, Vector)) => {
        BLAS.axpy(v._1, v._2, u._2)
        (u._1 + v._1, u._2)
      },
      combOp = (u1: (Double, DenseVector), u2: (Double, DenseVector)) => {
        BLAS.axpy(1.0, u2._2, u1._2)
        (u1._1 + u2._1, u1._2)
      }
    )
    aggregated
  }

  def aggregateUnlabeledData(data: RDD[Vector], learnedModel: NaiveBayesModel) : Array[(Double, Vector)] = {
    val numLabels   = learnedModel.labels.length
    val numFeatures = learnedModel.theta(0).length

    val posteriors: RDD[(Vector, Vector)] = data.mapPartitions(iter => iter.map(v => (learnedModel.predictProbabilities(v), v))).cache
    val aggregated =
      for (k <- 0 until numLabels) yield {
        val kPosteriors: RDD[(Double, Vector)] = posteriors.map( t => (t._1(k), t._2))
        aggregateData(kPosteriors, numFeatures)
      }
    posteriors.unpersist(true)
    aggregated.toArray
  }

  def estimateLabels(data: RDD[Vector], learnedModel: NaiveBayesModel) : RDD[(Double, Vector)] = {
    data.mapPartitions(iter => iter.map(v => (learnedModel.predict(v), v)))
  }


  def run(data: RDD[LabeledPoint]): NaiveBayesModel = {
    val labeledData   = data.filter(p => p.label > 0)
    val unlabeledData = data.filter(p => p.label == 0).map(p => p.features)

    // aggregate labeled data
    val labeledAggregated = this.aggregateLabeledData(labeledData)
    val labeledModel = this.trainer.run(labeledData)

    var model: NaiveBayesModel = labeledModel

    if (unlabeledData.count == 0) {
      return labeledModel
    }

    var newTheta = model.theta.clone
    var newPi    = model.pi.clone

    val labels      = model.labels
    val numFeatures = model.theta(0).length

    // EM
    var prevLikelihood = this.logLikelihood(model, labeledData, unlabeledData)
    var iter = 0
    while (iter < getMaxIter) {
      logInfo(s"#$iter" )
      // e-step
      val unlabeledAggregated = this.aggregateUnlabeledData(unlabeledData, model)

      // m-step
      // labeled
      labeledAggregated.foreach {
        case(label, (weight, sumTermFreq)) => {
          val labelIndex = label.toInt - 1
          var j = 0
          while (j < numFeatures) {
            newTheta(labelIndex)(j) =  sumTermFreq(j) + lambda
            j += 1
          }
          newPi(labelIndex) = weight + lambda
        }
      }

      // unlabeled
      var likelihood = 0
      unlabeledAggregated.zipWithIndex.foreach {
        case ((posterior, sumTermFreq), k) => {
          val label = labels(k)
          val labelIndex = label.toInt - 1
          var j = 0
          while (j < numFeatures) {
            newTheta(labelIndex)(j) += sumTermFreq(j)
            j += 1
          }
          newPi(labelIndex) += posterior
          //likelihood += math.log(posterior)
        }
      }

      // normalize and logfy
      newTheta = newTheta.map(ArrayUtil.normalize)
      newTheta = newTheta.map {
        probs => probs.map(math.log(_))
      }
      newPi    = ArrayUtil.normalize(newPi)
      newPi = newPi.map(math.log(_))

      // model re-construction
      model = new NaiveBayesModel(labels, newPi, newTheta, modelType)

      val lik = this.logLikelihood(model, labeledData, unlabeledData)
      val eps = math.abs((lik - prevLikelihood) / prevLikelihood)
      newTheta = labeledModel.theta.clone
      newPi    = labeledModel.pi.clone


      // debug
      println("#" + iter)
      println("prevLikelihood " + prevLikelihood)
      println("likelihood " + likelihood)
      println("eps " + eps)

      if (eps < tolerance) {
        return model
      }

      iter += 1
      prevLikelihood = lik
    }
    model
  }

  def logLikelihood(model: NaiveBayesModel, labeled: RDD[LabeledPoint], unlabeled: RDD[Vector]) : Double = {
    // labeled
    var lik = 0.0
    labeled.foreach {
      lp => {
        val label = lp.label
        val features = lp.features
        val p= model.jointProbabilities(features)(label.toInt-1)
        lik += math.log(p)
      }
    }
    // unlabeled
    unlabeled.foreach {
      v => {
        val ps = model.jointProbabilities(v)
        lik += math.log(ps.toArray.sum)
      }
    }
    lik
  }
}
