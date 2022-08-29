import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Modal from '@material-ui/core/Modal';
import Button from '@material-ui/core/Button';


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
  };
}

const useStyles = makeStyles((theme) => ({
  paper: {
    position: 'absolute',
    width: 400,
    backgroundColor: theme.palette.background.paper,
    border: '2px solid #000',
    boxShadow: theme.shadows[5],
    padding: theme.spacing(2, 4, 3),
  },
}));

export default function Tutorial() {
  const classes = useStyles();
  // getModalStyle is not a pure function, we roll the style only on the first render
  const [modalStyle] = React.useState(getModalStyle);
  const [open, setOpen] = React.useState(false);

  const handleOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const body = (
    <div style={modalStyle} className={classes.paper}>

        <h2>App Usage</h2> 
        <p>
        Welcome! The goal of this app to help the user label documents more quickly. We use clustering, machine label recommendations, and next document recommendations. 
        <br/>
        Documents are initially grouped by topic. A topic is represented by a list of words.
        <br/>
        Click on a document to open a detailed view. You can label it with an existing label or create a new one. You can also see metadata like the machine predicted label, and its confidence score.
        <br/>
        After a few documents are labelled, the app will recommend the next document to label. 
        <br/>
        Navbar: the "table view" lists the documents and metadata in a table, with the ability to sort and filter.
        </p>

    </div>
  );

  return (
    <div>
      <Button onClick={handleOpen} color="inherit">Help</Button>
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