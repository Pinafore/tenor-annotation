/* eslint-disable */
import React, { useState } from 'react';
import { Redirect, Link } from 'react-router-dom';

import {postRequest, getRequest} from '../utils';

import '../App.css';
import { createMuiTheme, withStyles, makeStyles, ThemeProvider, styled } from '@material-ui/core/styles';
import useStyles from '../Styles';


import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';

import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';

import Navbar from '../Components/Navbar'
import DocumentDetail from '../Components/DocumentDetail';
import DocumentList from '../Components/DocumentList';

import TextForm from '../Components/TextForm';
import DropdownForm from '../Components/DropdownForm.jsx';
import RenameForm from '../Components/RenameForm.jsx';

import LabelDetail from '../Components/LabelDetail.jsx';
import DocumentsByTopic from '../Components/DocumentsByTopic.jsx';

import CircularProgress from '@material-ui/core/CircularProgress';


// const Item = styled(Paper)(({ theme }) => ({
//     ...theme.typography.body2,
//     padding: theme.spacing(1),
//     textAlign: 'center',
//     color: theme.palette.text.secondary,
//   }));


class Dashboard extends React.Component {
    constructor(props) {
        super(props);

        this.labelDocument = this.labelDocument.bind(this);
        this.addLabel = this.addLabel.bind(this);
        this.deleteLabel = this.deleteLabel.bind(this);
        this.renameLabel = this.renameLabel.bind(this);
        this.refreshData = this.refreshData.bind(this);
        this.exportData = this.exportData.bind(this);

        this.state = {
            document_clusters: [],  
            documents: [],
            topics: [],   
            labels: [],
            highlightedDoc: null,
            stats: {},
            settings: {},
            docToHighlight: null,

            isLoading: false,
        }
    }

    async refreshData(sort_docs = 'uncertainty') {
        console.log('refreshing data...');
        this.setState({isLoading: true});

        // TODO: set interval to check on topic model
        // intervalID = setInterval(async () => {
        //     let result = await getRequest('/get_preprocessing_status');
        //     console.log('data status', result);
        //     setProcessingStatus(result);

        //     // if (processingStatus === 'none' || processingStatus === 'finished') {
        //     //     datasetStatus = <h4>Loaded Dataset: {dataset}</h4>
        //     // } 
        // }, 3000);
        // console.log('interval', intervalID)
        // setIntervalID(interval);

        const settings = await getRequest(`/get_settings`);
        sort_docs = settings['sort_docs_by'];
        const group_size = settings['num_top_docs_shown'];
        console.log('sort_docs', sort_docs);
        const res = await getRequest(`/get_document_clusters?group_size=${group_size}&sort_by=${sort_docs}`);
        const docToHighlight = res['doc_to_highlight'];
        console.log('docToHighlight', docToHighlight);
        const document_clusters = res['document_clusters'];
        console.log('document_clusters', document_clusters);

        
    
        const labels = await getRequest(`/get_labels`);
        // console.log('labels', labels);

        const stats = await getRequest(`/get_statistics`);
        // console.log('stats', stats);

        const documents_grouped_by_label = await getRequest(`/get_documents_grouped_by_label`);
        // console.log('documents_grouped_by_label', documents_grouped_by_label);

        // const settings = await getRequest(`/get_settings`);
        // console.log('settings', settings); 

        this.setState({
            document_clusters: document_clusters,
            labels: labels,
            stats: stats,
            documents_grouped_by_label: documents_grouped_by_label,
            settings: settings,
            docToHighlight: docToHighlight,
            isLoading: false
        });

    }

    async componentDidMount() {
        await this.refreshData();

        let docToHighlight = this.state.docToHighlight;
        let element = document.getElementById(docToHighlight);
        // highlight text, scroll to it
        console.log('docToHighlight', docToHighlight, element);
        if (element !== null) {
            element.style.border = "thick solid red";
            element.scrollIntoView({behavior: "smooth", block: "center"});
        }
        
    }

    async exportData() {
        let token = window.sessionStorage.getItem("token");
        fetch('/export_data', {
            method: 'GET',
            headers: {
                // 'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
        }).then((res) => { return res.blob(); }).then((data) => {
          var a = document.createElement("a");
          a.href = window.URL.createObjectURL(data);
          a.download = "exported_annotations.jsonl";
          a.click();
        });
    }

    
    // highlight next doc to label, scroll to it
    async labelDocument(doc_id, label) {
        console.log('label doc:', doc_id, label);
        console.log('doc element', document.getElementById(doc_id));
        document.getElementById(doc_id).style.color = "green";
        this.setState({isLoading: true});
        // force refresh page, not sure how to do this automatically
        // this.forceUpdate();
        const resp = await postRequest(`/label_document?doc_id=${doc_id}&label=${label}`);
        console.log('label doc response', resp);


        // refresh the documents
        await this.refreshData();
        // force refresh page, not sure how to do this automatically
        // this.forceUpdate();

        // scroll to next doc to label, highlight (regardless of active learning status - in no active learning case, next doc chosen randomly)
        
        const next_doc = this.state.docToHighlight;
        console.log('next_doc', next_doc);

        if (resp['status'] === 'active_learning_update') {
            console.log("active learning");
        } else {
            console.log('no active learning');
        }
        let element = document.getElementById(next_doc);
        console.log('next element', element);
        if (element !== null) {
            /*
            if (this.state.highlightedDoc) {
                console.log("in the remove border if block");
                this.state.highlightedDoc.style.borderColor = null;
            }
            */
            element.style.border = "thick solid red";
            element.scrollIntoView({behavior: "smooth", block: "center"});
            // remove border of previous element
            if (this.state.highlightedDoc) {
                console.log("in the remove border if block");
                this.state.highlightedDoc.style.borderColor = null;
            }
            this.setState({
                highlightedDoc: element
            });
        } else {
            console.log("couldn't find document!");
        }


    }

    async addLabel(label) {
        const resp = await postRequest(`/add_label?label=${label}`);

        const labels = await getRequest(`/get_labels`);
        const stats = await getRequest(`/get_statistics`);
        this.setState({
            labels: labels,
            stats: stats,
        });

    }

    async deleteLabel(label) {
        const resp = await postRequest(`/delete_label?label=${label}`);

        this.refreshData();
    }

    async renameLabel(oldLabel, newLabel) {
        const resp = await postRequest(`/rename_label?old_label=${oldLabel}&new_label=${newLabel}`);

        this.refreshData();
    }



    render() {

        if (window.sessionStorage.getItem("token") == null) {
            return <Redirect to="/register" />;
        }

        const { classes } = this.props;

        // const topics = this.state.topics.map((words, index) =>
        //         <ListItem key={index}>
        //             <ListItemText primary={words.join(', ')} />
        //         </ListItem>
        //     );
        
        const label_list = this.state.labels.map((label, index) =>
            // <ListItem key={label}>
            //     <ListItemText primary={label} />
            // </ListItem>

            <li key={index}>
                <LabelDetail label={label} documents={this.state.documents_grouped_by_label[label]}/>
            </li>
        );

        const num_docs = this.state.stats['num_docs'];
        const num_labelled_docs = this.state.stats['num_labelled_docs'];
        const num_labels = this.state.stats['num_labels'];
        const num_confident_predictions = this.state.stats['num_confident_predictions'];

        const statistics = <ul>
            <li>Documents labelled: {num_labelled_docs}/{num_docs}={(num_labelled_docs/num_docs).toFixed(2)}</li>
            {/* <li>Number of predictions with score >=0.8: {num_confident_predictions}</li> */}
            <li>Number of confident predictions: {num_confident_predictions}/{num_docs}={(num_confident_predictions/num_docs).toFixed(2)}</li>
            <li>Labels: {num_labels}</li>

        </ul>

        // loading bar
        let loadingBar;
        if (this.state.isLoading) {
            loadingBar = <div>
                Processing... <br/>
                <CircularProgress style={{margin: 20}}/>
            </div>;
        }

        


        return (

            <div className={classes.root}>
    
                <Navbar/>
    
                <div className={classes.body} style={{ maxWidth: 1500, margin: "auto", paddingTop: 100 }}>

                <Grid container spacing={2}>
                    
                    <Grid item xs={4} >
                        <div style={{position: "sticky", top: 100}}>

                    <h2>Welcome, {window.sessionStorage.getItem("username")}</h2>

                    {loadingBar}

                    <h2>Progress</h2>
                    {statistics}

                    <h2>Labels</h2>
                    <div style={{ 
                        maxHeight: 400, 
                        // height: 'fit-content',
                        overflow: "scroll", 
                        whiteSpace: "pre-wrap", 
                        textAlign: "left", 
                        // padding: 20
                        
                    }}>
                        <ul>
                        {label_list}
                        </ul>
                    </div>

                        {/* buttons to add, delete, rename labels */}
                        <TextForm text="New label" onSubmit={(label) => {this.addLabel(label)}}/>
                        <DropdownForm text="Delete label" labels={this.state.labels} onSubmit={(label) => {this.deleteLabel(label)}}/>
                        <RenameForm labels={this.state.labels} onSubmit={(oldLabel, newLabel) => {this.renameLabel(oldLabel, newLabel)}}/>
                        <Button variant="contained" color="primary" onClick={this.exportData}>
                            {/* <a href='/export_data' download>Export data</a> */}
                            Export Data
                        </Button>
                        {/* <Link to="/export_data">
                            <Button variant="contained" color="primary" disableElevation>
                                Export Data
                            </Button>
                        </Link> */}
                        
                    </div>
                    </Grid>

                    <Grid item xs={8}>
                        {/* <Button variant="contained" color="primary" onClick={this.exportData}>
                            Sort documents by...
                        </Button> */}
                        {/* <DropdownForm text="Sort documents by..." 
                            labels={['uncertainty','confidence']} 
                            onSubmit={(label) => { 
                                this.refreshData(label)}}/> */}

                        <DocumentsByTopic 
                            labels={this.state.labels} 
                            labelDocument={this.labelDocument}
                            useTopics={this.state.settings['use_topic_model']}
                            documentClusters={this.state.document_clusters}
                        />
                    </Grid>
                </Grid>

                


                </div>
            </div>
        )
    }
}

export default withStyles(useStyles)(Dashboard);
