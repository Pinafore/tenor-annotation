import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';

import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
// import { ListItem } from '@material-ui/core';



export default function DropdownForm(props) {
  const { classes } = props;
  const [label, setLabel] = React.useState('label');
  const [error, setError] = React.useState(false);

  const handleChange = (event) => {
    setLabel(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (label === '') {
      setLabel(label);
      setError(true);
    } else {
      setLabel(label);
      setError(false);
      props.onSubmit(label);
    }
  }


  return (

    <form onSubmit={props.onSubmit}
      noValidate
      autoComplete="off"
      style={{ "display": "flex", "alignItems": "center" }}
    >
      <Select
        labelId="demo-simple-select-label"
        id="demo-simple-select"
        value={label}
        label="Label"
        onChange={handleChange}
        style={{ width: 120 }}
        error={error}
      >
        {props.labels.map((label, index) =>
          <MenuItem value={label} key={index}>{label}</MenuItem>
        )}
      </Select>

      <div style={{ padding: 20 }}>
        <Button variant="contained" color="primary" onClick={(event) => handleSubmit(event)}>
          {props.text}
        </Button>
      </div>
    </form>
  );
}