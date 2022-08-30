// Home, upload data, start session

import React, { useState, useEffect } from 'react';
import {
    BrowserRouter as Router,
    Switch,
    Route,
    Link
} from "react-router-dom";
import { Redirect } from "react-router-dom";

import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';
import CircularProgress from '@material-ui/core/CircularProgress';

import { postRequest, getRequest, getScheduleInfo } from '../utils';

import '../App.css';
import { createMuiTheme, withStyles, makeStyles, ThemeProvider } from '@material-ui/core/styles';
import useStyles from '../Styles';

import Navbar from '../Components/Navbar';
import Settings from '../Components/Settings';

// import { Steps, Hints } from 'intro.js-react';
// import introJs from 'intro.js';
// import 'intro.js/introjs.css';


export default withStyles(useStyles)(function Home(props) {

    const [playerData, setPlayerData] = useState({});
    const [sessionStats, setSessionStats] = useState("Loading Session Statistics...");
    // const [settings, setSettings] = useState("none");
    const [dataset, setDataset] = useState();

    const [selectedFile, setSelectedFile] = useState();
	const [isFilePicked, setIsFilePicked] = useState(false);
	const [isProcessingData, setIsProcessingData] = useState(false);
	const [processingStatus, setProcessingStatus] = useState("");

	const [settings, setSettings] = useState("");
	const [intervalID, setIntervalID] = useState();
    // var intervalID;


    const reset = async () => {
        if (window.confirm("This will permanently reset all current work. Click OK to continue.")) {
            const resp = await postRequest(`/reset`);

            window.location.reload();
        }
    }

    const exportData = async () => {
        const resp = await postRequest(`/export_data`);
    }

    // for select file button
	const changeHandler = (event) => {
		setSelectedFile(event.target.files[0]);
        if (selectedFile) {
            setIsFilePicked(true);
        }
	};

    const startSession = async () => {
        console.log('starting session...')
        clearInterval(intervalID);
    }

    const usePreloadedData = async () => {
        if (window.confirm("This will permanently reset all current work. Click OK to continue.")) {

            postRequest(`/use_preloaded_data`).then(value => {
                // setIsProcessingData(false);
                setProcessingStatus('using preloaded data');

                setDataset('RAS guidance docs'); // TODO: remove hardcode
            });
        }
    }


    // upload data to server, process
	const handleSubmission = async (fileType, importAnnotations) => {

        if (!selectedFile) {
            alert("no file selected!");
            return
        }

		const formData = new FormData();

		formData.append('file', selectedFile);
        console.log('formData',formData);

        setIsProcessingData(true);

        // postRequest(`/upload_zip`, formData).then(resp => {
        // postRequest(`/upload_data_dummy`, formData)
        // postRequest(`/upload_data_dummy`, formData)
        postRequest(`/upload_file?file_type=${fileType}&import_annotations=${importAnnotations}`, formData)

        // if (format === "zip") {
        //     postRequest(`/upload_file?file_type=${fileType}&import_annotations=${importAnnotations}`, formData)
        // } else if (format === "csv") {
        //     postRequest(`/upload_file?file_type=${fileType}&import_annotations=${importAnnotations}`, formData)
        // }
        
        // check upload status
        var interval = setInterval(async () => {
            let result = await getRequest('/get_preprocessing_status');
            console.log('data status', result);
            setProcessingStatus(result);

            if (result === 'finished') {
                console.log("finished!");
                setIsProcessingData(false);
                clearInterval(interval);
            } 
        }, 3000);
        console.log('interval', interval);
        setIntervalID(interval);

	};
	

    const { classes } = props;


    // leaderboard = DataTable(rows, columns)
    // Similar to componentDidMount and componentDidUpdate:
    useEffect(() => {
        async function fetchData() {
            console.log('getting data...');

            const player_info = await getRequest(`/get_player_info?username=${window.sessionStorage.getItem("username")}`);
            console.log('player info', player_info);
            setPlayerData(player_info);

            let result = await getRequest('/get_preprocessing_status');
            console.log('data status', result);
            setProcessingStatus(result);

            const stats = await getRequest(`/get_statistics`);
            console.log('stats', stats);   
            setSessionStats(stats); 

            setDataset(stats['dataset_name']);
            console.log('dataset', dataset);

            const settings = await getRequest(`/get_settings`);
            console.log('settings', settings); 
            setSettings(settings);

            
            // check dataset status
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
            
        }
        // check if user is logged in
        if (window.sessionStorage.getItem("token") == null) {
            return
        } else {
            fetchData();

            // clear interval when we leave the page
            return function cleanup() {
                console.log('clearing interval...', intervalID)
                clearInterval(intervalID)
            };
        }
        
    }, []); // Or [] if effect doesn't need props or state


    // check if user is logged in
    if (window.sessionStorage.getItem("token") == null) {
        return <Redirect to="/register" />;
    }

    // function DataTable(rows, columns) {
    //     return (
    //         <div style={{ height: 400, width: '100%' }}>
    //             <DataGrid rows={rows} columns={columns} pageSize={5} />
    //         </div>
    //     );
    // }

    const start_button = <Link to="/">
                            <Button onClick={startSession} variant="contained" color="primary" className={classes.margin}>
                                Start Annotation Session!
                            </Button>
                        </Link>

    const preload_button = <button onClick={usePreloadedData}>Use sample data (RAS guidance docs)</button>
    // const preload_button = <Button onClick={usePreloadedData} variant="contained" color="primary" className={classes.margin}>
    //                             Use Preloaded Data
    //                         </Button>
    const reset_button = <Button onClick={reset} variant="contained" color="secondary" className={classes.margin}>
        Reset
    </Button>

    const export_button = <Button onClick={exportData} variant="contained" color="primary" className={classes.margin}>
        Export Data
    </Button>

    let datasetStatus;
    if (isProcessingData) {
        datasetStatus = <div>
            Processing data. This can take a few minutes... <br/>
            <CircularProgress style={{margin: 20}}/>
        </div>;
    } else if (processingStatus.includes('error')) {
        console.log('processingStatus', processingStatus)
        datasetStatus = <h4>Error!</h4>
    } 
    else if (!isProcessingData || processingStatus === 'none' || processingStatus === 'finished') {
        datasetStatus = <h4>Loaded Dataset: {dataset}</h4>
    } else {
        datasetStatus = <div>
            Processing data. This can take a few minutes... <br/>
            <CircularProgress style={{margin: 20}}/>
        </div>;
    }
    
    let num_docs = sessionStats['num_docs'];
    let num_labelled_docs = sessionStats['num_labelled_docs'];
    let num_labels = sessionStats['num_labels'];
    let num_confident_predictions = sessionStats['num_confident_predictions'];

    const stats = <ul>
        <li>Documents labelled: {num_labelled_docs}/{num_docs}</li>
        <li>Labels: {num_labels}</li>
        <li>Number of predictions with score >=0.8: {num_confident_predictions}</li>
    </ul>

    return (

        <div className={classes.root}>

            <Navbar/>

            <div className={classes.body} style={{ maxWidth: 1500, margin: "auto", marginTop: 50}}>

            <h2>Welcome, {window.sessionStorage.getItem("username")}</h2>

            <h3>Dataset Loader</h3>
            Processing status: {processingStatus}
            {datasetStatus}
            
            {/* <p>Custom data: please upload a properly formatted csv file. Must have the column "text"</p> */}
            <p>Custom data: please upload either a zipped directory of text files, or a properly formatted csv (see instructions).</p>

            <div>
			<input type="file" name="file" onChange={changeHandler} />
			{selectedFile ? (
				<div>
					<p>Filename: {selectedFile.name}</p>
					<p>Filetype: {selectedFile.type}</p>
					<p>Size in bytes: {selectedFile.size}</p>
					<p>
						lastModifiedDate:{' '}
						{selectedFile.lastModifiedDate.toLocaleDateString()}
					</p>
				</div>
			) : (
				<p></p>
			)}
			<div>
				{/* <button onClick={() => handleSubmission(false)}>Upload passages and train topic model</button> */}
                <button onClick={() => handleSubmission("zip", false)}>Upload zip file and train topic model</button>
			</div>
            <div>
                <button onClick={() => handleSubmission("csv", false)}>Upload csv and train topic model</button>
            </div>
            <div>
                <button onClick={() => handleSubmission("csv", true)}>Upload csv with labels and train topic model</button>
            </div>

            {preload_button}


            
		</div>
            
            {/* <ul>
                <li>use topic model: yes</li>
                <li>topic model: LDA</li>
                <li>number of topics: 20</li>
                <li>tfidf min threshold: </li>
                <li>tfidf max threshold: </li>
                <li>use active learning: yes</li>
                <li>classification model: Naive Bayes</li>
            </ul> */}
            <Settings />

            {/* <form onSubmit={this.handleSubmit}>
                <label>
                Settings
                <textarea value={this.state.value} onChange={this.handleChange} />
                </label>
                <input type="submit" value="Submit" />
            </form> */}

            {start_button} <br/>

            <h3>Valid data formats</h3>
            <ul>
                <li>Zipped directory of text files (.txt), each file should be a short (~200 words) passage.</li>
                <li>CSV file, with the columns "doc_id", "text", "label". Optional columns are "source".</li>

            </ul>


            <h3>Session Statistics</h3>
            {stats}

            {reset_button}
            {export_button}

            </div>
        </div>
    )
})