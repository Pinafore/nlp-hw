
qanta.train.json qanta.test.json qanta.dev.json:
	mkdir -p data
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.dev.2018.04.18.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.test.2018.04.18.json
	mv qanta.dev.2018.04.18.json data/qanta.dev.json	
	mv qanta.test.2018.04.18.json data/qanta.test.json
	mv qanta.train.2018.04.18.json data/qanta.train.json	
	
qanta.train.evidence.json qanta.dev.evidence.json qanta.test.evidence.json:
	mkdir -p data
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/evidence_docs_train.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/evidence_docs_dev.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/evidence_docs_test.json
	mv evidence_docs_dev.json data/qanta.dev.evidence.json	
	mv evidence_docs_test.json data/qanta.test.evidence.json
	mv evidence_docs_train.json data/qanta.train.evidence.json
	
qanta.train.evidence.text.json qanta.dev.evidence.text.json qanta.test.evidence.sent.text.json:
	mkdir -p data
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/evidence_docs_train_with_sent_text.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/evidence_docs_dev_with_sent_text.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/evidence_docs_test_with_sent_text.json
	
	mv evidence_docs_dev_with_sent_text.json data/qanta.dev.evidence.text.json	
	mv evidence_docs_test_with_sent_text.json data/qanta.test.evidence.text.json
	mv evidence_docs_train_with_sent_text.json data/qanta.train.evidence.text.json
