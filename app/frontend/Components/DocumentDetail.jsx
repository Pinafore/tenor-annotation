/* eslint-disable */
import React, { useState, useEffect } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Modal from '@material-ui/core/Modal';
import Box from '@material-ui/core/Box';

import Link from '@material-ui/core/Link';

import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
// import { ListItem } from '@material-ui/core';

import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';

import TextForm from '../Components/TextForm';
import DropdownForm from '../Components/DropdownForm.jsx';


function rand() {
  return Math.round(Math.random() * 20) - 10;
}

function getModalStyle() {
  const top = 50
  const left = 50

  return {
    top: `${top}%`,
    left: `${left}%`,
    transform: `translate(-${top}%, -${left}%)`,
    maxHeight: '80vh', 
    // width: '80vh',
    width: '90vh',

  };
}

const useStyles = makeStyles((theme) => ({
  paper: {
    position: 'absolute',
    // width: 400,
    backgroundColor: theme.palette.background.paper,
    border: '2px solid #000',
    boxShadow: theme.shadows[5],
    padding: theme.spacing(2, 4, 3),
  },
}));

export default function DocumentDetail(props) {
  const classes = useStyles();
  // getModalStyle is not a pure function, we roll the style only on the first render
  const [modalStyle] = React.useState(getModalStyle);
  const [open, setOpen] = React.useState(false);
  const [color, setColor] = React.useState("black");

  // const [label, setLabel] = React.useState('');

  // Similar to componentDidMount and componentDidUpdate:
  useEffect(() => {

    if (props.document['manual_label'] === "") {
      setColor("black");
    } else {
      setColor("green");
    }
  });

  // const handleChange = (event) => {
  //   setLabel(event.target.value);
  // };

  const handleOpen = () => {
    setOpen(true);

    var p = document.getElementById("previousPassage");
    console.log('previousPassage', p);
    // mainPassage.scrollIntoView();

  };

  const handleClose = () => {
    setOpen(false);
  };
  

  const body = (
    // className="content bordered" 
    <div style={modalStyle} className={classes.paper}>
        <h2>Passage Detail</h2> 
        <div style={{display: 'flex'}}>

          {/* passage text*/}
          <div style={{ 
              // minWidth: 300,
              width: 400,
              maxHeight: 400, 
              // height: 'fit-content',
              overflow: "scroll", 
              scrollTop: 100,

              whiteSpace: "pre-wrap", 
              textAlign: "left", 
              // padding: 20
            }}>

            {/* {props.document['previous_passage']} */}
            {/* <div id="mainPassage" style={{
              backgroundColor: "yellow",
              }}>
                *** Passage to label: ***
              {props.document['text']}
            </div> */}
            {/* {props.document['next_passage']} */}
            {props.document['text']}
          </div>
            
          {/* metadata */}
          <div style={{
            padding:20,
            width: 200,
            // wordWrap: "break-word",
            overflow: "hidden",
            textOverflow: "ellipsis",

            width: 400
            }}>

          <List>
            <ListItem>Passage ID: {props.document['doc_id']}</ListItem>
            <ListItem>Document title: {props.document['source']}</ListItem>
            <ListItem>manual label: {props.document['manual_label']}</ListItem>
            <ListItem>model prediction: {props.document['predicted_label']}</ListItem>
            <ListItem>prediction score: {props.document['prediction_score']}</ListItem>
            <ListItem>uncertainty score: {props.document['uncertainty_score']}</ListItem>
            <ListItem>largest topic percent: {props.document['dominant_topic_percent']}</ListItem>

            
          </List>


          <DropdownForm text="Submit label" labels={props.labels} 
            onSubmit={(label) => {handleClose(); props.onLabel(props.document['doc_id'], label)}}/>
          <TextForm text="New label" 
            onSubmit={(label) => {handleClose(); props.onLabel(props.document['doc_id'], label)}}/>
          </div>
        </div>

    </div>
  );


  return (
    <div>
      <Link onClick={handleOpen} 
        // color={props.colorMap[props.document['manual_label']]}
        style={{ color: color }}
        >
        {props.document['text'].slice(0,100).concat('...')}
        
      </Link>

      {/* pop up modal */}
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby="simple-modal-title"
        aria-describedby="simple-modal-description"
      >
        {body}
      </Modal>
    </div>
  );
}
