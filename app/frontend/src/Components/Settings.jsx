import React, { useState, useEffect } from 'react';

import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import Box from '@material-ui/core/Box';

import Button from '@material-ui/core/Button';

import { postRequest, getRequest, getScheduleInfo } from '../utils';

import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';

import '../App.css';
import { createMuiTheme, withStyles, makeStyles, ThemeProvider } from '@material-ui/core/styles';
import useStyles from '../Styles';


export default withStyles(useStyles)(function Settings(props) {

    let defaultSettings = `{
    "use_topic_model":true,
    "topic_model":"LDA",
    "num_topics":10,
    "tfidf_min":2,
    "tfidf_max":90,
    "use_active_learning":true,
    "classification model":"Naive Bayes",
    "num_top_docs_shown":-1,
    "sort_docs_by":"uncertainty",
    "sample_size": 500,
    "batch_update_freq": 3
}`;
	const [settingsText, setSettingsText] = useState("");
	const [settings, setSettings] = useState({});

    const { classes } = props;

    const handleChange = (event) => {
        // this.setState({value: event.target.value});
        setSettingsText(event.target.value);
      }
    
    const handleSubmit = async (event) => {
        alert('submitted: ' + settingsText);
        const serverSettings = await postRequest('/set_settings', settingsText);
        console.log('settings', serverSettings);
        setSettings(serverSettings);

        event.preventDefault();
      }

    
    // Similar to componentDidMount and componentDidUpdate:
    useEffect(() => {
        async function fetchData() {
            console.log('getting data...');

            const serverSettings = await getRequest(`/get_settings`);
            console.log('settings', serverSettings);
            setSettings(serverSettings);
            setSettingsText(JSON.stringify(serverSettings, null, 2));
        }
        fetchData();
    }, []); // Or [] if effect doesn't need props or state
    
    // console.log(Object.entries(settings))

    const settingsList = Object.entries(settings).map((line, index) =>
        // <ListItem key={index}>
        //     <ListItemText primary={line} />
        // </ListItem>
        <li key={index}>
            {line.join(': ')}
        </li>
    );

    return (

        <div className={classes.root}>
            {/* <div className={classes.body} style={{ maxWidth: 1500, margin: "auto" }}> */}

            <h2>Settings</h2>

            {/* <ul>
                <li>use topic model: yes</li>
                <li>topic model: LDA</li>
                <li>number of topics: 20</li>
                <li>tfidf min threshold: </li>
                <li>tfidf max threshold: </li>
                <li>use active learning: yes</li>
                <li>classification model: Naive Bayes</li>
            </ul> */}
            
            {/* <List>
            {settingsList}
            </List> */}
            <ul>
            {settingsList}
            </ul>

            <form onSubmit={handleSubmit}>
                <label>
                Edit settings (must be valid JSON) <br/><br/>
                <textarea value={settingsText} onChange={handleChange} style={{height: 200, width: 500}}/>
                </label>
                <br/>
                <input type="submit" value="set settings" />
            </form>

            <br/>

            {/* </div> */}
        </div>
    )
})
