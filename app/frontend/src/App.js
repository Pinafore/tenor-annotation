import * as React from "react";
import * as s from "./App.css";
import Dashboard from './Views/Dashboard';
import TableView from './Views/TableView';

// import Login from "./Login";
import Register from "./Views/DummyLogin";

// import Dashboard from './Views/Dashboard_intro';
import Home from './Views/Home';

import { BrowserRouter as Router, Switch, Route } from "react-router-dom";


function App() {
  console.log("App style "+s);
  return (
    <Router>
      <Switch>
        {/* <Route path="/login">
          <Login />
        </Route> */}
        <Route path="/register">
          <Register />
        </Route>
        <Route path="/home">
          <Home />
        </Route>
        <Route path="/table_view">
          <TableView />
        </Route>
        <Route path="/">
          <Dashboard />
        </Route>
      </Switch>
    </Router>
  );
}

export default App;