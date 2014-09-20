package com.wma.logistic;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

import org.apache.commons.io.FileUtils;


import Jama.Matrix;

public class MushroomTest {

	private static final String DATA_SET_PATH = "./mushroom";

	private static final double TRAINING_SIZE = 0.7;

	private static Matrix theta = null;

	public static void main(String[] args) throws IOException {
		List<String> lines = FileUtils.readLines(new File(DATA_SET_PATH));
		int totalSize = lines.size();
		Collections.shuffle(lines);
		int index = (int) ((totalSize-1)*TRAINING_SIZE);
		
		List<String> train = lines.subList(0, index);
		List<String> test = lines.subList(index, totalSize-1);
		// 获取第一行训练数据，得到feature的数量
		String sample = lines.get(0);
		int featureSize = sample.split("\t").length - 1;

		train(train, featureSize);
		test(test, featureSize);
	}

	private static void test(List<String> lines, int featureSize)
			throws IOException {

		double right = 0.0;
		double all = 0.0;

		// 一行一个事务
		for (String line : lines) {
			String[] n = line.split("\t");
			int length = n.length;
			// 一个事务所有属性的向量
			double[][] arr = new double[1][length];
			// 列向量，其中第一个元素为分类标签，在下面要全部换成1。
			for (int i = 0; i < length; i++) {
				arr[0][i] = Double.valueOf(n[i]);
			}
			arr[0][0] = 1D;
			
			int result;
			if (Integer.valueOf(n[0]) == 1) {
				result = 1;
			} else {
				result = 0;
			}

			int s = GradAscentModel.judge(new Matrix(arr), theta) ? 1 : 0;

			if (result == s) {
				right++;
			}
			all++;
		}
		System.out.println(right);
		System.out.println(all);
		System.out.println(right / all);
	}

	private static void train(List<String> lines, int featureSize)
			throws IOException {

		int dataSetSize = lines.size();

		double[][] trainingSet = new double[dataSetSize][featureSize + 1];
		double[][] labels = new double[dataSetSize][1];

		int index = 0;
		for (String line : lines) {
			String[] n = line.split("\t");
			int length = n.length;
			for (int i = 0; i < length; i++) {
				trainingSet[index][i] = Double.valueOf(n[i]);
			}
			trainingSet[index][0] = Double.valueOf(1);
			if (Integer.valueOf(n[0]) == 1) {
				labels[index][0] = Double.valueOf(1);
			} else {
				labels[index][0] = Double.valueOf(0);
			}
			index++;
		}

		Matrix dataSet = new Matrix(trainingSet);
		Matrix label = new Matrix(labels);
		// theta = GradAscentModel.gradAscent(dataSet, label);
		theta = GradDescentModel.gradDescent(dataSet, label);
		// theta = GradDescentModel.regularizedGradDescent(dataSet, label);

		double[][] arr = theta.getArray();
		int outterLength = arr.length;
		for (int i = 0; i < outterLength; i++) {
			int innerLength = arr[i].length;
			for (int j = 0; j < innerLength; j++) {
				System.out.println("theta:" + arr[i][j]);
			}
		}
	}
}
