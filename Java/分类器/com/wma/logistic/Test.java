package com.wma.logistic;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

public class Test {

	private static final String INPUT_PATH = "";
	private static final String OUTPUT_PATH = "";
	
	public static void main(String[] args) throws Exception{
		
		List<String> lines = FileUtils.readLines(new File(INPUT_PATH));
		List<String> result = new ArrayList<String>();
		
		for (String line:lines){
			String[] infos = line.split("\t");
			StringBuilder sb= new StringBuilder();
			sb.append(infos[11]);
			sb.append("\t");
			sb.append(infos[2]);
			sb.append("\t");
			sb.append(infos[4]);
			sb.append("\t");
			sb.append(infos[6]);
			sb.append("\t");
			sb.append(infos[8]);
			sb.append("\t");
			sb.append(infos[10]);
			
//			sb.append(infos[0]);
//			sb.append("\t");
//			sb.append(infos[1]);
//			sb.append("\t");
//			sb.append(infos[2]);
//			sb.append("\t");
//			sb.append(infos[4]);
//			sb.append("\t");
//			sb.append(infos[6]);
//			sb.append("\t");
//			sb.append(infos[8]);
//			sb.append("\t");
//			sb.append(infos[10]);
//			sb.append("\t");
//			sb.append(infos[11]);
			
			result.add(sb.toString());
		}
		
		FileUtils.writeLines(new File(OUTPUT_PATH),result);
		
	}
	
	
}
