#!/bin/bash
dims=(10 30 100 300);
epochs=(10 20);
wordNgrams=(1 2);
models_dir="../models/"
features_dir="../feature_vectors/"

for dim in ${dims[*]}; do
	for epoch in ${epochs[*]}; do
		for wordNgram in ${wordNgrams[*]}; do
			base_name="mincount8_dim${dim}_epoch${epoch}_lr02_wordNgrams${wordNgram}_lossns_neg10_thread12_t0000005_dropoutk4_mincountlabel20_bucket4000000";
			model_name="${models_dir}${base_name}"
			feature_name="${features_dir}${base_name}"
			cmd="/home/doru/sent2vec/fasttext sent2vec -input /home/doru/data/tweets_dataset.txt -output ${model_name} -minCount 8 -dim ${dim} -epoch ${epoch} -lr 0.2 -wordNgrams ${wordNgram} -loss ns -neg 10 -thread 12 -t 0.000005 -dropoutK 4 -minCountLabel 20 -bucket 4000000";
			echo $cmd;
			$cmd;
			echo "Done computing the model";
			/home/doru/sent2vec/fasttext print-sentence-vectors ${model_name}.bin < /home/doru/data/train_pos.txt > ${feature_name}_pos
			echo "Done computing pos vectors representations";
			/home/doru/sent2vec/fasttext print-sentence-vectors ${model_name}.bin < /home/doru/data/train_neg.txt > "${feature_name}_neg";
			echo "Done computing neg vectors representations";
			python /home/doru/src/main.py --pospath "${feature_name}_pos" --negpath "${feature_name}_neg";
		done
	done
done
