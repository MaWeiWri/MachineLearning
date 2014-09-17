package com.wma.bayes;

import java.util.ArrayList;
import java.util.List;

/**
 * 贝叶斯分类器输入对象
 * @author MaWei
 *
 */
public class BayesObject {
	
	int featureSize = 0;
	
	List<Integer> features = new ArrayList<Integer>();
	
	int results = -1;
	
	/**
	 * 贝叶斯分类器输入对象构造器
	 * @param featureSize 该分类器中输入feature数量
	 */
	public BayesObject(int featureSize){
		this.featureSize = featureSize;
	}
	
	/**
	 * 为输入对象添加feature的值
	 * @param feature feature的值
	 */
	public void addFeature(int feature){
		features.add(feature);
	}
	
	/**
	 * 检查输入对象feature是否完整
	 * @return 是否完整
	 */
	public boolean checkBayesObject(){
		if (featureSize==features.size()){
			return true;
		}else {
			return false;
		}
	}

	public void setResults(int results) {
		this.results = results;
	}

	public int getResults() {
		return results;
	}
}
