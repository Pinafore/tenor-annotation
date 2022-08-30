import React, { useState } from 'react';
import { Redirect } from "react-router-dom";

import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import Box from '@material-ui/core/Box';

import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';

import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';

import Navbar from '../Components/Navbar'

import {postRequest, getRequest, getScheduleInfo} from '../utils';

import '../App.css';
import { createMuiTheme, withStyles, makeStyles, ThemeProvider } from '@material-ui/core/styles';
import useStyles from '../Styles';

import DocumentDetail from '../Components/DocumentDetail';
import DocumentList from '../Components/DocumentList';

import TextForm from '../Components/TextForm';
import DropdownForm from '../Components/DropdownForm.jsx';
import RenameForm from '../Components/RenameForm.jsx';


// import Tabs from '@material-ui/core/Tabs';
// import Tab from '@material-ui/core/Tab';



class TableView extends React.Component {
    constructor(props) {
        super(props);

        this.labelDocument = this.labelDocument.bind(this);
        this.addLabel = this.addLabel.bind(this);
        this.deleteLabel = this.deleteLabel.bind(this);
        this.renameLabel = this.renameLabel.bind(this);

        this.state = {
            document_clusters: [],  
            documents: [],
            topics: [],   
            labels: [],
            // colors: ['red','blue','green'],
            colorMap: {'one':'red', 'two':'blue', 'three':'green', null:'black'},
            highlightedDoc: null,
            stats: {},

        }
    }

    async componentDidMount() {
        
        // const topics = await getRequest(`/get_topics`);
        // console.log('topics', topics);

        // const document_clusters = await getRequest(`/get_document_clusters`);
        // console.log('document_clusters', document_clusters);
        
        // const labels = await getRequest(`/get_labels`);
        // console.log('labels', labels);

        const documents = await getRequest(`/get_documents`);
        console.log('documents', documents);

        // const stats = await getRequest(`/get_statistics`);
        // console.log('stats', stats);

        this.setState({
            // topics: topics['topic_words'],
            // document_clusters: document_clusters,
            // labels: labels,
            documents: documents,
            // stats: stats,
        });

    }

    

    // scroll to next doc to label 
    async labelDocument(doc_id, label) {
        console.log('label doc:', doc_id, label)
        const resp = await postRequest(`/label_document?doc_id=${doc_id}&label=${label}`);
        console.log('label doc response', resp);

        // refresh the documents
        const document_clusters = await getRequest(`/get_document_clusters`);     
        console.log('first document', document_clusters[0]['documents'][0]);               
        this.setState({
            document_clusters: document_clusters
        });

        // scroll to next doc to label, highlight
        if (resp['status'] === 'active_learning_update') {
            const next_doc = resp['next_doc_id_to_label'];
            console.log('next_doc', next_doc)

            const element = document.getElementById(next_doc);
            // if element is null?
            console.log('next element', element)
            if (element !== null) {
                // element.style.borderColor = "red";
                element.style.border = "thick solid red";
                element.scrollIntoView();
            }
            
            
            // remove border of previous element
            // this.state.highlightedDoc.style.borderColor = null;
            this.setState({
                highlightedDoc: element
            });
            
        } else {
            console.log('no active learning')
        }

    }

    async addLabel(label) {
        const resp = await postRequest(`/add_label?label=${label}`);

        const labels = await getRequest(`/get_labels`);
        const stats = await getRequest(`/get_statistics`);
        console.log('labels', labels);
        this.setState({
            labels: labels,
            stats: stats,
        });

    }

    async deleteLabel(label) {
        const resp = await postRequest(`/delete_label?label=${label}`);

        const labels = await getRequest(`/get_labels`);
        const stats = await getRequest(`/get_statistics`);
        const document_clusters = await getRequest(`/get_document_clusters`);

        this.setState({
            document_clusters: document_clusters,
            labels: labels,
            stats: stats,
        });

    }

    async renameLabel(oldLabel, newLabel) {
        const resp = await postRequest(`/rename_label?old_label=${oldLabel}&new_label=${newLabel}`);
        const labels = await getRequest(`/get_labels`);
        // const stats = await getRequest(`/get_statistics`);
        const document_clusters = await getRequest(`/get_document_clusters`);

        this.setState({
            document_clusters: document_clusters,
            labels: labels,
            // stats: stats,
        });
    }

    render() {

        const { classes } = this.props;

        // const topics = this.state.topics.map((words, index) =>
        //         <ListItem key={index}>
        //             <ListItemText primary={words.join(', ')} />
        //         </ListItem>
        //     );
        
        // const label_list = this.state.labels.map((label, index) =>
        //     // <ListItem key={label}>
        //     //     <ListItemText primary={label} />
        //     // </ListItem>

        //     <ul>
        //         <li>{label}</li>
        //     </ul>
        // );

        // const document_clusters = this.state.document_clusters.map((row, index) =>
        //     <ListItem key={row['topic_words']} style={{borderBottom: "thin solid black"}}>
        //         <ListItemText primary={row['topic_words']} />
        //         {
        //             <List>{
        //             row['documents'].map((document, idx2) =>
        //                 <ListItem key={document.doc_id} id={document.doc_id}>
                            
        //                     <DocumentDetail 
        //                         document={document} 
        //                         labels={this.state.labels}
        //                         onLabel={(doc_id, label) => this.labelDocument(doc_id, label)}
        //                         colorMap={this.state.colorMap}
        //                     />
                        
        //                 </ListItem>
        //             )
        //             }</List>
        //         }
        //     </ListItem>
        // );
        
        // const num_docs = this.state.stats['num_docs'];
        // const num_labelled_docs = this.state.stats['num_labelled_docs'];
        // const num_labels = this.state.stats['num_labels'];
        // const num_confident_predictions = this.state.stats['num_confident_predictions'];

        // const statistics = <ul>
        //     <li>Documents labelled: {num_labelled_docs}/{num_docs}</li>
        //     <li>Labels: {num_labels}</li>
        //     <li>Number of predictions with score >=0.8: {num_confident_predictions}</li>

        // </ul>



        return (

            <div className={classes.root}>
    
                <Navbar/>
    
                <div className={classes.body} style={{ maxWidth: 1500, margin: "auto" }}>

                {/* <h2>Progress</h2>
                {statistics}

                <h2>Labels</h2>
                {label_list} */}


                <h2>Table View</h2>
                {/* {document_clusters} */}
                <DocumentList documents={this.state.documents}/>
    
                </div>
            </div>
        )
    }
}

export default withStyles(useStyles)(TableView);
