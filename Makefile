
qanta.train.json qanta.test.json qanta.dev.json:
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json
	mv qanta.train.2018.04.18.json qanta.train.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.dev.2018.04.18.json
	mv qanta.dev.2018.04.18.json qanta.dev.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.test.2018.04.18.json
	mv qanta.test.2018.04.18.json qanta.test.json
qanta.train.evidence.json qanta.dev.evidence.json qanta.test.evidence.json:
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/evidence_docs_train.json
	mv evidence_docs_train.json qanta.train.evidence.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/evidence_docs_dev.json
	mv evidence_docs_dev.json qanta.dev.evidence.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/evidence_docs_test.json
	mv evidence_docs_test.json qanta.test.evidence.json
qanta.train.evidence.text.json qanta.dev.evidence.text.json qanta.test.evidence.sent.text.json:
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/evidence_docs_train_with_sent_text.json
	mv evidence_docs_train_with_sent_text.json qanta.train.evidence.text.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/evidence_docs_dev.json
	mv evidence_docs_dev_with_sent_text.json qanta.dev.evidence.text.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/evidence_docs_test.json
	mv evidence_docs_test_with_sent_text.json qanta.test.evidence.text.json
