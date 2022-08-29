import React from 'react';
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
    width: '80vh',

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

export default function LabelDetail(props) {
  const classes = useStyles();
  // getModalStyle is not a pure function, we roll the style only on the first render
  const [modalStyle] = React.useState(getModalStyle);
  const [open, setOpen] = React.useState(false);

  // const [label, setLabel] = React.useState('');

  // const handleChange = (event) => {
  //   setLabel(event.target.value);
  // };

  const handleOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  // if (props.documents)
  
  const body = (
    // className="content bordered" 
    // className={classes.paper}
    <div style={modalStyle} className={classes.paper}>
        <h2>Label Detail</h2> 

        Number of documents with label: {props.documents.length}
        <div style={{ 
            maxHeight: 400, 
            // height: 'fit-content',
            overflow: "scroll", 
            whiteSpace: "pre-wrap", 
            textAlign: "left", 
            padding: 20
          }}>
            <ul>
                {props.documents.map((doc, index) => <li key={index}>
                    Document {doc['doc_id']}: {doc['text'].slice(0,100).concat('...')
                }</li>)}
            </ul>
        </div>


    </div>
  );

//   let color;
//   if (props.document['manual_label'] === "") {
//     color = "black";
//   } else {
//     color = "green";
//   }

  return (
    <div>
      <Link onClick={handleOpen} 
        // color={props.colorMap[props.document['manual_label']]}
        style={{ color: 'black' }}
        >
        {/* {props.document['text'].slice(0,100).concat('...')} */}
        {props.label}
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