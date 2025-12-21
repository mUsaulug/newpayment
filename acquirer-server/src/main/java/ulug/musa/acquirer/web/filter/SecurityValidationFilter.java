package ulug.musa.acquirer.web.filter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import ulug.musa.common.security.SecurityHeaders;
import ulug.musa.acquirer.security.validation.RequestSecurityService;
import ulug.musa.acquirer.security.validation.SecurityExceptionHandler;
import ulug.musa.acquirer.security.validation.SecurityValidationException;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Component
public class SecurityValidationFilter extends OncePerRequestFilter {

    private final RequestSecurityService securityService;
    private final SecurityExceptionHandler exceptionHandler;

    public SecurityValidationFilter(RequestSecurityService securityService,
                                    SecurityExceptionHandler exceptionHandler) {
        this.securityService = securityService;
        this.exceptionHandler = exceptionHandler;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String path = request.getRequestURI();

        if ("/ping".equals(path)) return true;
        return !path.startsWith("/api");
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {

        CachedBodyHttpServletRequest wrapped = new CachedBodyHttpServletRequest(request);

        Map<String, String> headers = new HashMap<>();
        headers.put(SecurityHeaders.TERMINAL_ID, request.getHeader(SecurityHeaders.TERMINAL_ID));
        headers.put(SecurityHeaders.NONCE, request.getHeader(SecurityHeaders.NONCE));
        headers.put(SecurityHeaders.TIMESTAMP, request.getHeader(SecurityHeaders.TIMESTAMP));
        headers.put(SecurityHeaders.SIGNATURE, request.getHeader(SecurityHeaders.SIGNATURE));

        String body = wrapped.bodyAsString();

        try {
            securityService.validate(headers, body);
            filterChain.doFilter(wrapped, response);
        } catch (SecurityValidationException e) {
            // Eksik header / invalid signature vb. -> 4xx dön
            exceptionHandler.write(response, 400, "SEC_VALIDATION_FAILED", e.getMessage());
        }
    }
}
