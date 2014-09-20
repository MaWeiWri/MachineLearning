package com.wma.logistic;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.io.FileUtils;

import Jama.Matrix;

public class LogisticRegressionModel {

	private static final String DATA_SET_PATH = "";

	private static final double TRAINING_SIZE = 0.8;

	private static final int FEATURE_SIZE = 5;

	private static final int GRADIENT_ASCENT = 1;
	private static final int GRADIENT_DESCENT = 2;
	private static final int REGULARIZED_GRADIENT_DECENT = 3;

	private static int algorithm = REGULARIZED_GRADIENT_DECENT;

	private Matrix dataMatrix = null;

	private Matrix label = null;

	private Matrix theta = null;

	private double positivePositive = 0.0;
	private double negativePositive = 0.0;
	private double positiveNegative = 0.0;
	private double negativeNegative = 0.0;
	private double positive = 0.0;
	private double negative = 0.0;

	private double right = 0.0;
	private double all = 0.0;

	public static void main(String[] args) throws IOException {
		LogisticRegressionModel im = new LogisticRegressionModel();
		File[] files = new File(DATA_SET_PATH).listFiles();
//		do {
			im.remainTest(files);
//		} while (!im.needStop());
//		 im.initThetaTest(files);
	}

	private boolean needStop() {
		double p1 = positivePositive / positive;
		double p2 = negativePositive / negative;
		initStatistics();
		if (p1 > 0.7 && p2 > 0.7) {
			return true;
		}
		return false;
	}

	private void initStatistics() {
		dataMatrix = null;
		label = null;
		theta = null;
		positivePositive = 0.0;
		negativePositive = 0.0;
		positiveNegative = 0.0;
		negativeNegative = 0.0;
		positive = 0.0;
		negative = 0.0;
		right = 0.0;
		all = 0.0;
	}

	public void initThetaTest(File[] files) throws IOException {
		List<String> lines = new ArrayList<String>();
		for (File file : files) {
			lines.addAll(FileUtils.readLines(file));
		}
		double[][] init = { { -6 }, { 12 }, { 5 }, { 8 }, { 5 } };
		theta = new Matrix(init);
		test(lines);
	}

	public void crossTest(File[] files) throws IOException {
		int fileIndex = (int) (Math.random() * files.length);
		List<String> train = (FileUtils.readLines(files[fileIndex]));
		Collections.shuffle(train);

		System.out.println("Training Data:" + files[fileIndex].getName());
		// 读取训练数据集，将其转换为矩阵dataSet和label
		toMatrix(train);
		train();

		fileIndex = (int) (Math.random() * files.length);
		List<String> test = (FileUtils.readLines(files[fileIndex]));
		System.out.println("Test Data:" + files[fileIndex].getName());
		Collections.shuffle(test);
		test(test);
	}

	public void remainTest(File[] files) throws IOException {
		List<String> lines = new ArrayList<String>();
		for (File file : files) {
			lines.addAll(FileUtils.readLines(file));
		}

		int totalSize = lines.size();
		Collections.shuffle(lines);
		int index = (int) ((totalSize - 1) * TRAINING_SIZE);

		List<String> train = lines.subList(0, index);
		List<String> test = lines.subList(index, totalSize - 1);
		// 读取训练数据集，将其转换为矩阵dataSet和label
		toMatrix(train);
		train();
		test(test);
	}

	private void toMatrix(List<String> lines) {

		int dataSetSize = lines.size();

		// 构造m*n维训练数据矩阵，其中m为训练样本数量，n为feature数量+1
		// theta=[A,B,C]' dataSet=[1,x1,x2]
		// z=Ax0+Bx1+Cx2，其中x0=1
		double[][] dataSet = new double[dataSetSize][FEATURE_SIZE + 1];
		double[][] labels = new double[dataSetSize][1];

		// 行索引
		int index = 0;
		for (String line : lines) {
			// 一行记录按\t切分
			String[] n = line.split("\t");
			// 读取的原始数据中从index2 开始为feature
			// 训练数据矩阵中从index1 开始为feature
			for (int i = 0; i < FEATURE_SIZE; i++) {
				dataSet[index][i + 1] = Double.valueOf(n[i + 2]);
			}
			// 训练数据矩阵中index0 设为1(x0=1)
			dataSet[index][0] = Double.valueOf(1);
			// 读取label标记
			labels[index][0] = Double.valueOf(n[FEATURE_SIZE + 2]);
			index++;
		}

		dataMatrix = new Matrix(dataSet);
		label = new Matrix(labels);
	}

	private void test(List<String> lines) throws IOException {

		// 一行一个事务
		for (String line : lines) {
			String[] n = line.split("\t");
			// 一个事务所有属性的列向量
			double[][] arr = new double[1][FEATURE_SIZE + 1];
			// 列向量，从n[2]开始为数据集中的feature，在下面要全部换成1。
			for (int i = 0; i < FEATURE_SIZE; i++) {
				arr[0][i + 1] = Double.valueOf(n[i + 2]);
			}
			// 训练数据矩阵中第一个元素为1，即x0=1
			arr[0][0] = 1D;

			double y = Double.valueOf(n[FEATURE_SIZE + 2]);

			int result = GradAscentModel.judge(new Matrix(arr), theta) ? 1 : 0;

			
			statistic(result, y);
			// if (result == s) {
			// } else {
			// System.out.println(line);
			// }
		}
		double[][] arr = theta.getArray();
		int outterLength = arr.length;
		for (int i = 0; i < outterLength; i++) {
			int innerLength = arr[i].length;
			for (int j = 0; j < innerLength; j++) {
				System.out.println("theta: " + arr[i][j]);
			}
		}
		System.out.println("样本数:  1:" + positive + "\t0:" + negative + "\tAll:"
				+ all);
		System.out.println("正确分类数/训练集总数:" + right + "/" + all);
		System.out.println("分类正确率:" + right / all);
		System.out.println("P(result=1|y=1)=" + (int) positivePositive + "/"
				+ (int) positive + "=" + positivePositive / positive);
		System.out.println("P(result=0|y=0)=" + (int) negativeNegative + "/"
				+ (int) negative + "=" + negativeNegative / negative);
		System.out.println("P(result=0|y=1)=" + (int) positiveNegative + "/"
				+ (int) positive + "=" + positiveNegative / positive);
		System.out.println("P(result=1|y=0)=" + (int) negativePositive + "/"
				+ (int) negative + "=" + negativePositive / negative);
	}

	private void train() throws IOException {
		if (algorithm == GRADIENT_ASCENT) {
			theta = GradAscentModel.gradAscent(dataMatrix, label);
		} else if (algorithm == GRADIENT_DESCENT) {
			theta = GradDescentModel.gradDescent(dataMatrix, label);
		} else if (algorithm == REGULARIZED_GRADIENT_DECENT) {
			theta = GradDescentModel.regularizedGradDescent(dataMatrix, label);
		}
	}

	private void statistic(int result, double y) {
		if (y == 1) {
			positive++;
		} else {
			negative++;
		}
		// 如果分类正确，则right++
		if (result == y) {
			right++;
			// 分类为1，被判为1
			if (result == 1) {
				positivePositive++;
			} else {
				// 分类为0，被判为0
				negativeNegative++;
			}
		}
		// 分类错误
		else {
			// 分类为0，被判为1
			if (result == 1) {
				negativePositive++;
			} else {
				// 分类为1，被判为0
				positiveNegative++;
			}
		}
		all++;
	}
}
