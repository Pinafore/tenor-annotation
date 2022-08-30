import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';

import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
// import { ListItem } from '@material-ui/core';
import TextField from '@material-ui/core/TextField';



export default function RenameForm(props) {
  // const classes = useStyles();
  const { classes } = props;
  const [label, setLabel] = React.useState('label');
  const [newLabel, setNewLabel] = React.useState('');

  const handleChange = (event) => {
    setLabel(event.target.value);
  };
  const handleNewLabelChange = (event) => {
    setNewLabel(event.target.value);
  };

  return (

    <form onSubmit={props.onSubmit} className={props.root} noValidate autoComplete="off"
      style={{ "display": "flex", "alignItems": "center" }}>

      <Select
        labelId="demo-simple-select-label"
        id="demo-simple-select"
        value={label}
        label="Label"
        onChange={handleChange}
        style={{ width: 120, marginRight: 20 }}
      >
        {props.labels.map((label, index) =>
          <MenuItem value={label} key={index}>{label}</MenuItem>
        )}
      </Select>

      <TextField
        id="outlined-name"
        label="New label"
        value={newLabel}
        onChange={handleNewLabelChange}
      />

      <div style={{ padding: 20 }}>
        <Button variant="contained" color="primary" onClick={() => props.onSubmit(label, newLabel)}>
          Rename label
        </Button>
      </div>
    </form>

  );
}