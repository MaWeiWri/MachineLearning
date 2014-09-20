package com.wma.bayes;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;


/**
 * 朴素贝叶斯分类器 能够被序列化与反序列化以支持增量式训练与使用
 * 
 * @author MaWei
 * 
 */
public class NaiveBayes implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6960205280295385071L;

	/**
	 * 存储P(y=1)和P(y=0)的概率
	 */
	double py0 = 0;
	double py1 = 0;

	/**
	 * 贝叶斯分类器中输入feature的size
	 */
	private int featureSize = 0;

	/**
	 * 存储y=0事务计数
	 */
	private double y0 = 0;
	/**
	 * 存储y=1事务计数
	 */
	private double y1 = 0;

	/**
	 * 存储xn,y=0事务计数 外层Map中，key:Integer为feature标号(index)的值,value:该Index下的xi各值的计数
	 * 内层Map中，key为xi的值，value为对应值的计数
	 */
	private HashMap<Integer, HashMap<Integer, Integer>> xy0 = new HashMap<Integer, HashMap<Integer, Integer>>();

	/**
	 * 存储xn,y=1事务计数 外层Map中，key:Integer为feature标号(index)的值,value:该Index下的xi各值的计数
	 * 内层Map中，key为xi的值，value为对应值的计数
	 */
	private HashMap<Integer, HashMap<Integer, Integer>> xy1 = new HashMap<Integer, HashMap<Integer, Integer>>();

	/**
	 * 存储P(xi|y=0)的概率
	 * 外层Map中，key:Integer为feature标号(index)的值,value:该Index下的P(xi|y=0)
	 * 内层Map中，key为xi的值，value为对应概率
	 */
	private HashMap<Integer, HashMap<Integer, Double>> pxy0 = new HashMap<Integer, HashMap<Integer, Double>>();

	/**
	 * 存储P(xi|y=1)的概率
	 * 外层Map中，key:Integer为feature标号(index)的值,value:该Index下的P(xi|y=1)
	 * 内层Map中，key为xi的值，value为对应概率
	 */
	private HashMap<Integer, HashMap<Integer, Double>> pxy1 = new HashMap<Integer, HashMap<Integer, Double>>();

	/**
	 * 构造器，需要指定分类器中feature的数量
	 * 
	 * @param featureSize
	 */
	public NaiveBayes(int featureSize) {
		this.featureSize = featureSize;

		/**
		 * 初始化所有Map
		 */
		for (int i = 0; i < featureSize; i++) {
			xy0.put(i, new HashMap<Integer, Integer>());
			xy1.put(i, new HashMap<Integer, Integer>());
			pxy0.put(i, new HashMap<Integer, Double>());
			pxy1.put(i, new HashMap<Integer, Double>());
		}

	}

	/**
	 * 为分类器添加训练样本
	 * 
	 * @param bo
	 *            输入训练样本
	 */
	public void addTrainingData(BayesObject bo) {
		// 检查输入数据的feature数量是否与分类器指定的一致
		if (bo.featureSize != this.featureSize) {
			System.out.println("Numbers of Features in Training Data Error!");
			return;
		}
		// 首先检查训练数据的分类结果
		int result = bo.getResults();

		if (result == 0) {
			// y0计数
			y0++;
		} else if (result == 1) {
			// y1计数
			y1++;
		} else {
			System.out.println("The Label of Input Data Error!");
			return;
		}

		// 遍历输入数据的所有feature
		for (int i = 0; i < featureSize; i++) {
			// 取feature的值
			int xi = bo.features.get(i);
			// 取计数器对象
			HashMap<Integer, Integer> xiy0 = xy0.get(i);
			HashMap<Integer, Integer> xiy1 = xy1.get(i);
			// 初始化计数器
			if (!xiy0.containsKey(xi)) {
				xiy0.put(xi, 0);
			}
			if (!xiy1.containsKey(xi)) {
				xiy1.put(xi, 0);
			}

			if (result == 0) {
				// 如果第i维输入为xi，则对xi的计数+1用于计算P(xi|y=0)
				int temp = xiy0.get(xi);
				xiy0.put(xi, temp + 1);
			} else {
				// 如果第i维输入为xi，则对xi的计数+1用于计算P(xi|y=0)
				int temp = xiy1.get(xi);
				xiy1.put(xi, temp + 1);
			}
		}
	}

	/**
	 * 对朴素贝叶斯模型进行训练
	 * 
	 * @param type
	 *            模型训练方式
	 * @type 为1时使用极大似然估计进行训练
	 * @type 为2时使用拉普拉斯平滑进行训练
	 */
	public void train(int type) {
		// 计算P(y=0)概率
		// 计算P(y=1)概率
		if (type == 0) {
			py0 = y0 / (y1 + y0);
			py1 = y1 / (y1 + y0);
		} else if (type == 1) {
			py0 = (y0 + 1) / (y1 + y0 + 2);
			py1 = (y1 + 1) / (y1 + y0 + 2);
		} else {
			System.out.println("Train");
		}

		// 计算P(xn|y)概率
		for (int i = 0; i < featureSize; i++) {
			// 存放P(xi=0|y=0),P(xi=1|y=0),P(xi=0|y=1),P(xi=1|y=1)
			Map<Integer, Integer> tempY0 = xy0.get(i);
			Set<Integer> keysY0 = tempY0.keySet();
			int size0 = keysY0.size();
			Map<Integer, Double> tempPY0 = pxy0.get(i);
			for (int key : keysY0) {
				double p = 1;
				if (type == 0) {
					p = tempY0.get(key) / y0;
				} else if (type == 1) {
					p = (tempY0.get(key) + 1) / (y0 + size0);
				}
				tempPY0.put(key, p);
			}
			Map<Integer, Integer> tempY1 = xy1.get(i);
			Set<Integer> keysY1 = tempY1.keySet();
			int size1 = keysY1.size();
			Map<Integer, Double> tempPY1 = pxy1.get(i);
			for (int key : keysY1) {
				double p = 1;
				if (type == 0) {
					p = tempY1.get(key) / y1;
				} else if (type == 1) {
					p = (tempY1.get(key) + 1) / (y1 + size1);
				}
				tempPY1.put(key, p);
			}
		}
	}

	/**
	 * 使用训练模型进行预测
	 * 
	 * @param bo
	 *            输入数据
	 * @return 返回结果0或者1
	 */
	public int predict(BayesObject bo) {
		// 检查输入数据的feature数量是否与分类器指定的一致
		if (bo.featureSize != featureSize) {
			System.out.println("Numbers of Features in Training Data Error!");
			return -1;
		}

		double result_0 = calculateY0(bo.features);
		double result_1 = calculateY1(bo.features);

		if (result_0 > result_1) {
			return 0;
		} else {
			return 1;
		}
	}

	/**
	 * 计算P(y=0)*P(x1|y=0)*P(x2|y=0)*P(x3|y=0)....
	 * 
	 * @param features
	 *            输入feature
	 * @return
	 */
	private double calculateY0(List<Integer> features) {
		// 初始化结果为P(y=0)
		double p = py0;
		for (int i = 0; i < featureSize; i++) {
			int feature = features.get(i);
			// 获取P(xi|y=0)
			double temp = pxy0.get(i).get(feature);
			// 相乘得P(y=0)*P(xi=feature|y=0)*P(xi=feature|y=0)....
			p *= temp;
		}
		return p;
	}

	/**
	 * 计算P(y=1)*P(x|y=1)
	 * 
	 * @param features
	 *            输入feature
	 * @return
	 */
	private double calculateY1(List<Integer> features) {
		// 初始化结果为P(y=0)
		double p = py1;
		for (int i = 0; i < featureSize; i++) {
			int feature = features.get(i);
			// 获取P(xi|y=0)
			double temp = pxy1.get(i).get(feature);
			// 相乘得P(y=0)*P(xi=feature|y=0)*P(xi=feature|y=0)....
			p *= temp;
		}
		return p;
	}

	/**
	 * 将训练模型序列化，以便增量式训练与使用
	 * 
	 * @param path
	 *            序列化输出路径
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	synchronized public boolean save(String path) throws FileNotFoundException,
			IOException {
		ObjectOutputStream oos = null;
		try {
			oos = new ObjectOutputStream(new FileOutputStream(path));
			oos.writeObject(this);
		} finally {
			oos.close();
		}
		return true;
	}

	/**
	 * 将训练模型反序列化，以便增量式训练与使用
	 * 
	 * @param path
	 *            序列化输入路径
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static NaiveBayes load(String path) throws FileNotFoundException,
			IOException, ClassNotFoundException {
		ObjectInputStream ois = null;
		NaiveBayes nb = null;
		try {
			ois = new ObjectInputStream(new FileInputStream(path));
			nb = (NaiveBayes) ois.readObject();
		} finally {
			ois.close();
		}
		return nb;
	}
	
	public String toString(){
		StringBuffer sb = new StringBuffer();
		sb.append("==========================================================\n");
		sb.append("Naive Bayes Classifier's feature size is ");
		sb.append(featureSize);
		sb.append("\n");
		sb.append("==========================================================\n");
		return sb.toString();
	}

}
