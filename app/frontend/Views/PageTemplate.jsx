// Home, start game, leaderboard

import React, { useState, useEffect } from 'react';
import {
    BrowserRouter as Router,
    Switch,
    Route,
    Link
} from "react-router-dom";

import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import Box from '@material-ui/core/Box';

import Button from '@material-ui/core/Button';
import { Redirect } from "react-router-dom";


import Navbar from '../Components/Navbar'

import { postRequest, getRequest, getScheduleInfo } from '../utils';

import '../App.css';
import { createMuiTheme, withStyles, makeStyles, ThemeProvider } from '@material-ui/core/styles';
import useStyles from '../Styles';

// import { Steps, Hints } from 'intro.js-react';
// import introJs from 'intro.js';
// import 'intro.js/introjs.css';

import { DataGrid } from '@material-ui/data-grid';



export default withStyles(useStyles)(function Home(props) {

    const [playerData, setPlayerData] = useState({});
    const [playerStats, setPlayerStats] = useState("Loading Session Statistics...");
    const [dataset, setDataset] = useState('RAS Annotations')

    const { classes } = props;


    // leaderboard = DataTable(rows, columns)
    // Similar to componentDidMount and componentDidUpdate:
    useEffect(() => {
        async function fetchData() {
            console.log('getting data...');

            const player_info = await getRequest(`/get_player_info?username=${window.sessionStorage.getItem("username")}`);
            console.log('player info', player_info);
            setPlayerData(player_info);
            
        }
        // check if user is logged in
        if (window.sessionStorage.getItem("token") == null) {
            return
        }
        fetchData();
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

    const start_button = <Link to="/play">
            <Button variant="contained" color="primary" className={classes.margin}>
                Start Annotation Session!
            </Button>
        </Link>

    
    
    return (

        <div className={classes.root}>

            <Navbar/>

            <div className={classes.body} style={{ maxWidth: 1500, margin: "auto" }}>

            <h2>Welcome, {window.sessionStorage.getItem("username")}</h2>

            <h2>Loaded Dataset: {dataset}</h2>
            {start_button}

            <h2>Session Details</h2>
            {playerStats}

                <Grid container spacing={1}
                    bgcolor="background.paper"
                >
                    <Grid item xs={9} style={{ 'justifyContent': 'flex-start' }}>
                    </Grid>
                </Grid>
            </div>
        </div>
    )
})