import numpy as np
import pickle
from loading_input import *
from sklearn.neighbors import KDTree

def get_queries_dict(filename):
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("feature Loaded.")
		return queries

DATABASE_VECTORS_FILENAME = "database_img_feat_00642214.pickle"
QUERY_VECTORS_FILENAME = "query_img_feat_00642214.pickle"
#result output
output_file = "result_img_trans_mono_left_00240080.txt"
#load feature
DATABASE_VECTORS = get_queries_dict(DATABASE_VECTORS_FILENAME)
QUERY_VECTORS = get_queries_dict(QUERY_VECTORS_FILENAME)
#load label
QUERY_FILE= 'generate_queries/oxford_evaluation_query.pickle'
QUERY_SETS= get_sets_dict(QUERY_FILE)

for m in range(len(QUERY_SETS)):
	print(len(QUERY_VECTORS[m]))
	print(len(QUERY_SETS[m]))
	if len(QUERY_SETS[m]) != len(QUERY_VECTORS[m]):
		print("not equal")


def get_recall(m, n):
	global DATABASE_VECTORS
	global QUERY_VECTORS

	database_output= DATABASE_VECTORS[m]
	queries_output= QUERY_VECTORS[n]

	print(len(queries_output))
	database_nbrs = KDTree(database_output)

	num_neighbors=25
	recall=[0]*num_neighbors

	top1_similarity_score=[]
	one_percent_retrieved=0
	threshold=max(int(round(len(database_output)/100.0)),1)

	num_evaluated=0
	for i in range(len(queries_output)):
		true_neighbors= QUERY_SETS[n][i][m]
		if(len(true_neighbors)==0):
			continue
		num_evaluated+=1
		distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=num_neighbors)
		for j in range(len(indices[0])):
			if indices[0][j] in true_neighbors:
				if(j==0):
					similarity= np.dot(queries_output[i],database_output[indices[0][j]])
					top1_similarity_score.append(similarity)
				recall[j]+=1
				break
				
		if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors))))>0:
			one_percent_retrieved+=1
			
	if num_evaluated == 0:
		return -1,-1,-1
	one_percent_recall=(one_percent_retrieved/float(num_evaluated))*100
	recall=(np.cumsum(recall)/float(num_evaluated))*100
	print(recall)
	#print(np.mean(top1_similarity_score))
	print(one_percent_recall)
	return recall, top1_similarity_score, one_percent_recall 

def get_similarity(m, n):
	global DATABASE_VECTORS
	global QUERY_VECTORS

	database_output= DATABASE_VECTORS[m]
	queries_output= QUERY_VECTORS[n]

	threshold= len(queries_output)
	print(len(queries_output))
	database_nbrs = KDTree(database_output)

	similarity=[]
	for i in range(len(queries_output)):
		distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=1)
		for j in range(len(indices[0])):
			q_sim= np.dot(q_output[i], database_output[indices[0][j]])
			similarity.append(q_sim)
	average_similarity=np.mean(similarity)
	average_similarity= 1
	print(average_similarity)
	return average_similarity
	
	
def main():
	for i in range(len(DATABASE_VECTORS)):
		print("database feature shape",i,DATABASE_VECTORS[i].shape)
	
	for i in range(len(QUERY_VECTORS)):
		print("query feature shape",i,QUERY_VECTORS[i].shape)
		
	#convert feature to target format
	recall= np.zeros(25)
	count = 0
	similarity=[]
	one_percent_recall=[]
	
	#compute recall
	for m in range(len(QUERY_SETS)):
		for n in range(len(QUERY_SETS)):
			if(m==n):
				continue
			pair_recall, pair_similarity, pair_opr = get_recall(m, n)
			if(pair_opr == -1):
				continue
			recall+=np.array(pair_recall)
			count+=1
			one_percent_recall.append(pair_opr)
			for x in pair_similarity:
				similarity.append(x)
			
	print()
	ave_recall=recall/count
	print(ave_recall)
	
	#print(similarity)
	average_similarity= np.mean(similarity)
	print(average_similarity)
	
	print("Average Similarity:\n")
	ave_one_percent_recall= np.mean(one_percent_recall)
	
	
	print("Average Top 1% Recall:\n")
	print(ave_one_percent_recall)
	
	#filename=RESULTS_FOLDER +'average_recall_oxford_netmax_sg(finetune_conv5).txt'
	with open(output_file, "w") as output:
		output.write("Average Recall @N:\n")
		output.write(str(ave_recall))
		output.write("\n\n")
		output.write("Average Similarity:\n")
		output.write(str(average_similarity))
		output.write("\n\n")
		output.write("Average Top 1% Recall:\n")
		output.write(str(ave_one_percent_recall))

if __name__ == "__main__":
	main()
	