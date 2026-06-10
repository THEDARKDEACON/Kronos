import time
import threading
import uuid
import logging
from typing import Callable, Dict, Any, Optional

try:
    import quickfix as fix
    QUICKFIX_AVAILABLE = True
except ImportError:
    QUICKFIX_AVAILABLE = False
    class fix:
        Application = object


class AlpacaFixEngine(fix.Application if QUICKFIX_AVAILABLE else object):
    """
    QuickFIX Application for Alpaca FIX Protocol Integration.
    Handles Logon, Logoff, NewOrderSingle, and ExecutionReport messages.
    """
    def __init__(self, password: str, execution_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        super().__init__()
        self.session_id = None
        self.password = password
        self.execution_callback = execution_callback
        self.logger = logging.getLogger("AlpacaFIX")
        self.is_logged_in = False
        
        # Track pending orders mapping ClOrdID to actual symbols/details
        self.pending_orders = {}

    def onCreate(self, sessionID):
        self.session_id = sessionID
        self.logger.info(f"FIX Session Created: {sessionID}")

    def onLogon(self, sessionID):
        self.is_logged_in = True
        self.logger.info(f"FIX Logon Successful: {sessionID}")

    def onLogout(self, sessionID):
        self.is_logged_in = False
        self.logger.info(f"FIX Logout: {sessionID}")

    def toAdmin(self, message, sessionID):
        # Inject password into Logon (MsgType=A)
        try:
            msg_type = fix.MsgType()
            message.getHeader().getField(msg_type)
            if msg_type.getValue() == fix.MsgType_Logon:
                message.setField(fix.Password(self.password))
                message.setField(fix.ResetSeqNumFlag(True))
                # Some platforms require RawData/RawDataLength for password in FIX 4.2
                # But typically Password (tag 554) works for Alpaca
        except Exception as e:
            self.logger.error(f"Error in toAdmin: {e}")

    def toApp(self, message, sessionID):
        # Outbound application messages (e.g. NewOrderSingle)
        pass

    def fromAdmin(self, message, sessionID):
        # Inbound admin messages (e.g. Heartbeats, Reject)
        pass

    def fromApp(self, message, sessionID):
        # Inbound application messages (e.g. ExecutionReport)
        try:
            msg_type = fix.MsgType()
            message.getHeader().getField(msg_type)
            
            if msg_type.getValue() == fix.MsgType_ExecutionReport:
                self._handle_execution_report(message)
                
        except Exception as e:
            self.logger.error(f"Error in fromApp: {e}")

    def _handle_execution_report(self, message):
        """Parse ExecutionReport (MsgType=8) and trigger callback."""
        try:
            clOrdID = fix.ClOrdID()
            message.getField(clOrdID)
            
            execType = fix.ExecType()
            message.getField(execType)
            
            ordStatus = fix.OrdStatus()
            message.getField(ordStatus)
            
            symbol = fix.Symbol()
            message.getField(symbol)
            
            side = fix.Side()
            message.getField(side)
            
            report = {
                "clOrdID": clOrdID.getValue(),
                "execType": execType.getValue(),
                "ordStatus": ordStatus.getValue(),
                "symbol": symbol.getValue(),
                "side": "BUY" if side.getValue() == fix.Side_BUY else "SELL",
            }
            
            if message.isSetField(fix.LastShares()):
                lastShares = fix.LastShares()
                message.getField(lastShares)
                report["last_shares"] = lastShares.getValue()
                
            if message.isSetField(fix.LastPx()):
                lastPx = fix.LastPx()
                message.getField(lastPx)
                report["last_px"] = lastPx.getValue()
                
            self.logger.info(f"Execution Report Received: {report}")
            
            if self.execution_callback:
                self.execution_callback(report)
                
        except Exception as e:
            self.logger.error(f"Error parsing Execution Report: {e}")

    def send_order(self, symbol: str, qty: float, side: str, order_type: str = "MARKET", limit_price: float = None) -> str:
        """
        Build and send a NewOrderSingle (MsgType=D) message.
        Returns the ClOrdID.
        """
        if not QUICKFIX_AVAILABLE:
            raise ImportError("QuickFIX is not installed.")
            
        if not self.is_logged_in:
            self.logger.warning("Attempted to send order while FIX session is not logged in.")
            return None
            
        clOrdID = str(uuid.uuid4())[:16] # 16 char max for ClOrdID is safe
        
        msg = fix.Message()
        header = msg.getHeader()
        header.setField(fix.BeginString("FIX.4.2"))
        header.setField(fix.MsgType(fix.MsgType_NewOrderSingle))
        
        msg.setField(fix.ClOrdID(clOrdID))
        msg.setField(fix.HandlInst('1')) # Auto execution private
        msg.setField(fix.Symbol(symbol))
        msg.setField(fix.Side(fix.Side_BUY if side.upper() == "BUY" else fix.Side_SELL))
        msg.setField(fix.OrderQty(qty))
        
        if order_type.upper() == "MARKET":
            msg.setField(fix.OrdType(fix.OrdType_MARKET))
        elif order_type.upper() == "LIMIT":
            msg.setField(fix.OrdType(fix.OrdType_LIMIT))
            if limit_price:
                msg.setField(fix.Price(limit_price))
        
        msg.setField(fix.TimeInForce(fix.TimeInForce_DAY))
        
        self.pending_orders[clOrdID] = {"symbol": symbol, "qty": qty, "side": side}
        
        fix.Session.sendToTarget(msg, self.session_id)
        self.logger.info(f"Sent NewOrderSingle: {clOrdID} for {qty} {symbol}")
        
        return clOrdID
