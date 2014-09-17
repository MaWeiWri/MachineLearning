package com.wma.logistic;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;

import Jama.Matrix;



/**
 * LogisticRegression训练类
 * @author MaWei
 *
 */
public class GradAscentModel {

	private static final double E = Math.E;
	/**
	 * 梯度上升迭代次数
	 */
	private static final int ASCENT_COUNTS = 5000;
	/**
	 * 每次学习的step
	 */
	private static final double ALPHA = 0.003;

	public static void main(String[] args) throws IOException {

		test();
	}

	/**
	 * 使用梯度上升法计算权值矩阵
	 * 
	 * @param data
	 *            数据元素，一行一条记录
	 * @param label
	 *            样本标记，与元素对应
	 * @return 参数矩阵
	 */
	public static Matrix gradAscent(Matrix data, Matrix label) {
		int featureLength = data.getArray()[0].length;// 获取元素个数
		
//		Matrix theta = new Matrix(new double[featureLength][1]);// 初始化theta，为列矩阵
		Matrix theta = Matrix.random(featureLength, 1);// 初始化theta，随机选择初值，为列矩阵
		
		for (int i = 0; i < ASCENT_COUNTS; i++) {
			//训练结果
			Matrix h = sigmoid(data.times(theta));
			//训练误差为label - h
			Matrix error = label.minus(h);
			//
			Matrix derivative = data.transpose().times(error);
			theta.plusEquals(derivative.times(ALPHA));
		}
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
	
	public static boolean judge(Matrix input,Matrix theta){
		input.times(theta);
		double h=input.times(theta).getArray()[0][0];
		if (h>0.5){
			return true;
		}else {
			return false;
		}
	}

	private static void test() throws IOException {
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
		Matrix result = gradAscent(dataSet,label);
		
		double[][] arr = result.getArray();
		int outterLength = arr.length;
		for (int i = 0; i < outterLength; i++) {
			int innerLength = arr[i].length;
			for (int j = 0; j < innerLength; j++) {
				System.out.println(arr[i][j]);
			}
		}
	}
}
