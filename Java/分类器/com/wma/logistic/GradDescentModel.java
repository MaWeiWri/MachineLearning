package com.wma.logistic;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;

import Jama.Matrix;

public class GradDescentModel {

	private static final double E = Math.E;
	/**
	 * 梯度上升迭代次数
	 */
	private static final int DESCENT_COUNTS = 5000;
	/**
	 * 初始化theta
	 */
	private static final double INIT_THETA = 10;
	/**
	 * 每次学习的step
	 */
	private static final double ALPHA = 0.01;
	/**
	 * Regularize时使用的lambda
	 */
	private static final double LAMBDA = 0.01;

	private static double cost = 10000D;

	public static void main(String[] args) throws IOException {
		test();
	}

	/**
	 * 使用梯度上升法计算权值矩阵
	 * 
	 * @param data
	 *            数据元素，一行一条记录
	 * @param label
	 *            样本标记，与元素对应,为列矩阵
	 * @return 参数矩阵
	 */
	public static Matrix gradDescent(Matrix data, Matrix label) {
		int featureLength = data.getArray()[0].length;// 获取元素个数
		int m = data.getArray().length;// 获取事务个数

		// Matrix theta = new Matrix(new double[featureLength][1]);//
		// 初始化theta，为列矩阵
		Matrix theta = Matrix.random(featureLength, 1);//
		// 初始化theta，选择初值，为列矩阵
		// initTheta(theta);

		for (int i = 0; i < DESCENT_COUNTS; i++) {
			// 训练结果
			Matrix h = sigmoid(data.times(theta));
			// 训练误差为 h - label
			Matrix error = h.minus(label);
			//
			checkCost(h, label, m);
			Matrix derivative = data.transpose().times(error);
			theta.minusEquals(derivative.times(ALPHA / m));
		}
		System.out.println("Optimum Cost:" + cost);
		return theta;
	}

	public static Matrix regularizedGradDescent(Matrix data, Matrix label) {
		int featureLength = data.getArray()[0].length;// 获取元素个数
		int m = data.getArray().length;// 获取事务个数

		// Matrix theta = new Matrix(new double[featureLength][1]);//
		// 初始化theta，为列矩阵
		Matrix theta = Matrix.random(featureLength, 1);// 初始化theta，随机选择初值，为列矩阵
		// 初始化theta，选择初值，为列矩阵
		initTheta(theta);
		// 初始化矩阵，令第一个元素为0，表示不对第一个theta进行regularization
		Matrix selector = ones(featureLength, 1);
		selector.getArray()[0][0] = 0;

		for (int i = 0; i < DESCENT_COUNTS; i++) {
			// 训练结果
			Matrix h = sigmoid(data.times(theta));
			// 训练误差为 h - label
			Matrix error = h.minus(label);
			//
			Matrix derivative = data.transpose().times(error);

			Matrix regularElement = selector.arrayTimes(theta);
			regularElement.timesEquals(LAMBDA);

			checkRegularizedCost(h, label, theta, LAMBDA, m);

			derivative.minusEquals(regularElement);

			theta.minusEquals(derivative.times(ALPHA / m));

		}
		System.out.println("Optimum Cost:" + cost);
		return theta;
	}

	/**
	 * 对矩阵中所有元素求sigmoid,返回新的矩阵，不改变原矩阵
	 * 
	 * @param m
	 *            输入的矩阵
	 * @return 输出矩阵
	 */
	private static Matrix sigmoid(Matrix m) {
		double[][] arr = m.getArrayCopy();
		int outterLength = arr.length;
		for (int i = 0; i < outterLength; i++) {
			int innerLength = arr[i].length;
			for (int j = 0; j < innerLength; j++) {
				double temp = 1 + Math.pow(E, -arr[i][j]);
				arr[i][j] = 1 / temp;
			}
		}
		return new Matrix(arr);
	}

	private static Matrix log(Matrix m) {
		double[][] arr = m.getArrayCopy();
		int outterLength = arr.length;
		for (int i = 0; i < outterLength; i++) {
			int innerLength = arr[i].length;
			for (int j = 0; j < innerLength; j++) {
				double temp = Math.log(arr[i][j]);
				arr[i][j] = temp;
			}
		}
		return new Matrix(arr);
	}

	private static boolean checkCost(Matrix h, Matrix label, int m) {
		double k = getCost(h, label, m);
		// if (cost < k) {
		// System.out.println("Previous cost:" + cost);
		// System.out.println("New cost:" + k);
		// return false;
		// }
		System.out.println("New cost:" + k);
		cost = k;
		return true;
	}

	private static boolean checkRegularizedCost(Matrix h, Matrix label,
			Matrix theta, double lambda, int m) {
		double k = getRegularizedCost(h, label, theta, lambda, m);
		if (cost < k) {
			System.out.println("Previous cost:" + cost);
			System.out.println("New cost:" + k);
			return false;
		}
		System.out.println("New cost:" + k);
		cost = k;
		return true;
	}

	/**
	 * 计算cost function 的值
	 * 
	 * @param h
	 *            theta'*X
	 * @param label
	 *            训练数据集分类
	 * @param m
	 *            训练数据数目
	 * @return cost function 的值
	 */
	private static double getCost(Matrix h, Matrix label, int m) {
		Matrix m1 = log(h.transpose()).times(label);
		Matrix m2 = ones(h.getRowDimension(), h.getColumnDimension());
		Matrix m3 = log(m2.minus(h));
		Matrix y = m2.minus(label);
		Matrix m4 = y.transpose().times(m3);
		double k = m1.get(0, 0) + m4.get(0, 0);
		return -k / m;
	}

	private static double getRegularizedCost(Matrix h, Matrix label,
			Matrix theta, double lambda, int m) {
		Matrix m1 = log(h.transpose()).times(label);
		Matrix m2 = ones(h.getRowDimension(), h.getColumnDimension());
		Matrix m3 = log(m2.minus(h));
		Matrix y = m2.minus(label);
		Matrix m4 = y.transpose().times(m3);
		double k = m1.get(0, 0) + m4.get(0, 0);

		// 初始化矩阵，令第一个元素为0，表示不对第一个theta进行regularization
		Matrix selector = ones(theta.getRowDimension(), 1);
		selector.getArray()[0][0] = 0;
		Matrix regularizedElement = theta.arrayTimes(selector);
		regularizedElement = regularizedElement.transpose().times(
				regularizedElement);

		double temp = regularizedElement.get(0, 0);
		temp = temp * lambda / (2 * m);

		return temp + (-k / m);
	}

	/**
	 * 计算regularized cost function 的值
	 * 
	 * @param h
	 *            theta'*X
	 * @param label
	 *            训练数据集分类
	 * @param theta
	 *            本次训练结果theta
	 * @param lambda
	 *            regularization rate
	 * @param m
	 *            训练数据数目
	 * @return regularized cost function 的值
	 */
	private static double getCost(Matrix h, Matrix label, Matrix theta,
			double lambda, int m) {
		Matrix m1 = log(h.transpose()).times(label);
		Matrix m2 = ones(h.getRowDimension(), h.getColumnDimension());
		Matrix m3 = log(m2.minus(h));
		Matrix y = m2.minus(label);
		Matrix m4 = y.transpose().times(m3);
		double k = m1.get(0, 0) + m4.get(0, 0);
		return -k / m;
	}

	private static Matrix ones(int m, int n) {
		double[][] arr = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				arr[i][j] = 1;
			}
		}
		return new Matrix(arr);
	}

	public static boolean judge(Matrix input, Matrix theta) {
		input.times(theta);
		double h = input.times(theta).getArray()[0][0];
		if (h > 0.5) {
			return true;
		} else {
			return false;
		}
	}

	private static void test() throws IOException {
		System.out.println(Math.log(0));
		List<String> lines = FileUtils.readLines(new File("./testSet.txt"));

		double[][] trainingSet = new double[100][3];
		double[][] labels = new double[100][1];
		int index = 0;
		for (String line : lines) {
			String[] n = line.split("\t");
			trainingSet[index][0] = Double.valueOf(1);
			trainingSet[index][1] = Double.valueOf(n[0]);
			trainingSet[index][2] = Double.valueOf(n[1]);
			labels[index][0] = Double.valueOf(n[2]);
			index++;
		}

		Matrix dataSet = new Matrix(trainingSet);
		Matrix label = new Matrix(labels);
		Matrix result = gradDescent(dataSet, label);

		double[][] arr = result.getArray();
		int outterLength = arr.length;
		for (int i = 0; i < outterLength; i++) {
			int innerLength = arr[i].length;
			for (int j = 0; j < innerLength; j++) {
				System.out.println(arr[i][j]);
			}
		}
	}

	private static void initTheta(Matrix theta) {
		double[][] tempTheta = theta.getArray();
		int length = tempTheta.length;
		for (int i = 1; i < length; i++) {
			tempTheta[i][0] = Math.random() * INIT_THETA;
		}

//		tempTheta[0][0] = -1;
//		tempTheta[1][0] = 1;
//		tempTheta[2][0] = 2;
//		tempTheta[3][0] = 3;
//		tempTheta[4][0] = 1;
//		tempTheta[5][0] = 2;

	}
}
