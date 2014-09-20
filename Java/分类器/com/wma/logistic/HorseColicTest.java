package com.wma.logistic;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;


import Jama.Matrix;

public class HorseColicTest {

	private static Matrix theta = null;

	public static void main(String[] args) throws IOException {

		train();
		test();

	}

	private static void test() throws IOException{
		List<String> lines = FileUtils.readLines(new File("./horseColicTest.txt"));
		
		double right = 0.0;
		double all = 0.0;
		
		for (String line:lines){
			String[] n = line.split("\t");
			int length = n.length;
			double[][] arr = new double[1][length-1];
			for (int i = 0 ;i < length-1;i++){
				arr[0][i] = Double.valueOf(n[i]);
			}
			int result = Integer.valueOf(n[length-1]);
			
			int s = GradAscentModel.judge(new Matrix(arr), theta)?1:0;
			
			if (result==s){
				right++;
			}
			all++;
		}
		System.out.println(right);
		System.out.println(all);
		System.out.println(right/all);
	}

	private static void train() throws IOException {
		List<String> lines = FileUtils.readLines(new File(
				"./horseColicTraining.txt"));

		double[][] trainingSet = new double[299][21];
		double[][] labels = new double[299][1];

		int index = 0;
		for (String line : lines) {
			String[] n = line.split("\t");
			int length = n.length;
			for (int i = 0; i < length - 1; i++) {
				trainingSet[index][i] = Double.valueOf(n[i]);
				trainingSet[index][0] = Double.valueOf(1);
			}
			labels[index][0] = Double.valueOf(n[length - 1]);
			index++;
		}

		Matrix dataSet = new Matrix(trainingSet);
		Matrix label = new Matrix(labels);
//		theta = GradAscentModel.gradAscent(dataSet, label);
		theta = GradDescentModel.gradDescent(dataSet, label);
//		theta = GradDescentModel.regularizedGradDescent(dataSet, label);

//		double[][] arr = theta.getArray();
//		int outterLength = arr.length;
//		for (int i = 0; i < outterLength; i++) {
//			int innerLength = arr[i].length;
//			for (int j = 0; j < innerLength; j++) {
//				System.out.println("theta:" + arr[i][j]);
//			}
//		}
	}

}
