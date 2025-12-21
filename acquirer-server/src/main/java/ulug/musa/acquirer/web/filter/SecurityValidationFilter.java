package ulug.musa.acquirer.web.filter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import ulug.musa.acquirer.config.SecurityProperties;
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
    private final SecurityProperties.Headers headers;

    public SecurityValidationFilter(RequestSecurityService securityService,
                                    SecurityExceptionHandler exceptionHandler,
                                    SecurityProperties securityProperties) {
        this.securityService = securityService;
        this.exceptionHandler = exceptionHandler;
        this.headers = securityProperties.headers();
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
        headers.put(this.headers.terminalId(), wrapped.getHeader(this.headers.terminalId()));
        headers.put(this.headers.nonce(), wrapped.getHeader(this.headers.nonce()));
        headers.put(this.headers.timestamp(), wrapped.getHeader(this.headers.timestamp()));
        headers.put(this.headers.signature(), wrapped.getHeader(this.headers.signature()));

        String body = wrapped.bodyAsString();

        try {
            securityService.validate(headers, body);
            filterChain.doFilter(wrapped, response);
        } catch (SecurityValidationException e) {
            // Eksik header / invalid signature vb. -> 4xx dön
            exceptionHandler.write(response, e.status().value(), "SEC_VALIDATION_FAILED", e.getMessage());
        }
    }
}
