# Overview
In this repository you will find a Python implementation of KitNET; an online anomaly detector, based on an ensemble of autoencoders. From,

*Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai, "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", Network and Distributed System Security Symposium 2018 (NDSS'18)*

This repo contains the anomaly detector only. For the full network intrusion detection system from the paper, please see https://github.com/ymirsky/Kitsune-py.

# How to use
Create a feature set from a PCAP
1) In NetExtractor convert benign pcap to tsv
	* $ ./pcap2tsv <benign_pcap> <output_tsv>
		* i.e. ./pcap2tsv benign.pcap dataset
2) Extract features using NetExtractor
	* $ python3 main.py <tsv_from_prev_step> <output_feature_file>
		* i.e. python3 main.py dataset.tsv features.csv

Train your model
1) in Kitnet-py folder, train model (using a csv extracted from a pcap with benign traffic)
	* $ python3 example.py train <feature_file_from_prev_step> <location_to_save_trained_model>
		* i.e. python3 example.py train features.csv trained_model.pkl

Execute your model
1) in Kitnet-py folder execute on a csv extracted from a malicious or benign pcap
	* $ python3 example.py execute <feature_file_csv> <location_of_saved_model>
		* i.e. $ python3 example.py execute mirai3.csv trained_model.pkl