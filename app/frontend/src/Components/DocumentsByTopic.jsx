import React, { useState, useEffect } from 'react';
import Box from '@material-ui/core/Box';
import { postRequest, getRequest } from '../utils';

import Grid from '@material-ui/core/Grid';
import DocumentDetail from '../Components/DocumentDetail';

import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';

export default function DocumentsByTopic(props) {
    const [documents, setDocuments] = useState('');

    const handleChange = (event) => {
        setDocuments(event.target.value);
    };


    useEffect(() => {
        const fetchData = async () => {

            const settings = await getRequest(`/get_settings`);
            let group_size = settings['num_top_docs_shown'];
            let documents;
            let useTopics = settings['use_topic_model'];
            if (useTopics) {
                documents = props.documentClusters;
                console.log('documents', documents);
            } else {
                documents = await getRequest(`/get_documents?limit=100`);
            }

            let document_clusters;
            if (useTopics) {
                document_clusters = <Grid container spacing={2}>
                    <Grid item xs={4}>Topic words</Grid>
                    <Grid item xs={8}>Passages</Grid>
                    {
                        documents.map((row, index) =>
                            <div key={index} style={{ display: 'flex', border: 'thin solid gray', padding: '10px', width: '100%' }}>
                                <Grid item xs={4} key={row['topic_words']}
                                    style={{ height: '100%' }}>
                                    Top topic words:
                                    <br />
                                    <b>{row['topic_words']}</b>
                                    <br /><br />
                                    <div style={{ verticalAlign: 'middle' }}>Predicted topic label: {row['topic_label']}</div>
                                    <br />
                                    Number of labelled documents: {row['num_labelled_docs']}/{row['num_docs']}
                                    <br />
                                </Grid>
                                <Grid item xs={8}>
                                    {
                                        <List>{
                                            row['documents'].map((document, idx2) =>
                                                <ListItem key={document.doc_id} id={document.doc_id}>

                                                    <DocumentDetail
                                                        document={document}
                                                        labels={props.labels}
                                                        onLabel={(doc_id, label) => props.labelDocument(doc_id, label)}
                                                    />

                                                </ListItem>
                                            )
                                        }</List>
                                    }
                                </Grid>
                            </div>
                        )
                    }
                </Grid>

            } else {
                document_clusters = <Grid container spacing={2}>

                    <List>{

                        documents.map((document, idx2) =>
                            <ListItem key={document.doc_id} id={document.doc_id}>

                                <DocumentDetail
                                    document={document}
                                    labels={props.labels}
                                    onLabel={(doc_id, label) => props.labelDocument(doc_id, label)}
                                />

                            </ListItem>
                        )
                    }</List>
                </Grid>
            }

            setDocuments(document_clusters);
        };
        fetchData();
    }, [props]);


    return (
        <Box sx={{ minWidth: 120 }}>

            {documents}
        </Box>
    );
}
